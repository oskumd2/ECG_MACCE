
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from functools import reduce
from typing import Union, Tuple, MutableSequence, List, Optional

from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Module, Conv1d, BatchNorm1d, LeakyReLU

# from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
import warnings
import random
import os

class FinalModel(nn.Module):
    def __init__(self, block_size, block_depth, block_layers, hidden_size, kernel_num):
        super().__init__() 
        self.is_train = True
        self.num_classes = 1
        self.max_len = 1152

        self.block_size = block_size
        self.block_depth = block_depth
        self.block_layers = block_layers
        self.hidden_size = hidden_size
        self.return_hidden = False  

        self.blocks = nn.Sequential(
            Block(1, block_size, 2, block_depth, kernel_num),
            Block(block_size, block_size * 2, 2, block_depth + 1, kernel_num),
            Block(block_size * 2, block_size * 4, 2, block_depth + 2, kernel_num),
        )
        if block_layers==4:
            self.blocks.add_module(f"plus_block",Block(block_size * 4, block_size * 4, 1, block_depth + 3, kernel_num))
        self.blocks.add_module(f"plus_conv",nn.Conv2d(block_size * 4, 128, kernel_size=1))

        if self.is_train == True:
            self.mask = False
        else:
            self.mask = False
        self.embd_dim = 128
        self.transformer = ConvTransformerBackbone(n_in=128,
                                                   n_embd=self.embd_dim,
                                                   n_head=4,
                                                   n_embd_ks=5,
                                                   max_len=self.max_len,
                                                   mha_win_size=[19]*6 + [-1]*3,
                                                   arch=(2, 2, 8),
                                                   use_abs_pe=False)
        self.linearN = nn.Sequential(nn.Linear(60, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 1))
        self.linear1 = nn.Linear(self.embd_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size+2, self.num_classes) # !CONCAT! original 128

    def forward(self, x,a,g):  # !CONCAT!
        x = x.unsqueeze(1)
        x = self.blocks(x)

        x = F.upsample(x, size=(12, 1152), mode='bilinear') # F.upsample is now deprecated
        x = self.transformer(x)

        x = self.linearN(x.flatten(-2)).squeeze(-1)
        hidden = F.leaky_relu(self.linear1(x))
        x = torch.cat([hidden,a,g], dim=1) # !CONCAT!
        x = self.linear2(x)

        if self.return_hidden:
            return x, hidden
        return x

class FinalModel_embeddings(nn.Module):
    def __init__(self, block_size, block_depth, block_layers, hidden_size, kernel_num):
        super().__init__() #super(FinalModel, self).__init__()
        self.is_train = True
        self.num_classes = 1
        self.max_len = 1152

        self.block_size = block_size
        self.block_depth = block_depth
        self.block_layers = block_layers
        self.hidden_size = hidden_size
        self.return_hidden = False  # Add flag to control hidden state return

        self.blocks = nn.Sequential(
            Block(1, block_size, 2, block_depth, kernel_num),
            Block(block_size, block_size * 2, 2, block_depth + 1, kernel_num),
            Block(block_size * 2, block_size * 4, 2, block_depth + 2, kernel_num),
        )
        if block_layers==4:
            self.blocks.add_module(f"plus_block",Block(block_size * 4, block_size * 4, 1, block_depth + 3, kernel_num))
        self.blocks.add_module(f"plus_conv",nn.Conv2d(block_size * 4, 128, kernel_size=1))

        if self.is_train == True:
            self.mask = False
        else:
            self.mask = False
        self.embd_dim = 128
        self.transformer = ConvTransformerBackbone(n_in=128,
                                                   n_embd=self.embd_dim,
                                                   n_head=4,
                                                   n_embd_ks=5,
                                                   max_len=self.max_len,
                                                   mha_win_size=[19]*6 + [-1]*3,
                                                   arch=(2, 2, 8),
                                                   use_abs_pe=False)
        self.linearN = nn.Sequential(nn.Linear(60, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 1))
        self.linear1 = nn.Linear(self.embd_dim, hidden_size)
        #self.dropout = nn.Dropout(0.4)
        self.linear2 = nn.Linear(hidden_size+770, self.num_classes) # !CONCAT! original 128

    def forward(self, x,a,g,e):  # !CONCAT!
        x = x.unsqueeze(1)
        x = self.blocks(x)

        x = F.upsample(x, size=(12, 1152), mode='bilinear') # F.upsample is now deprecated
        x = self.transformer(x)

        x = self.linearN(x.flatten(-2)).squeeze(-1)
        hidden = F.leaky_relu(self.linear1(x))
        x = torch.cat([hidden,a,g,e], dim=1) # !CONCAT!
        x = self.linear2(x)

        if self.return_hidden:
            return x, hidden
        return x
    
def apply_layer(layer_input: Tensor,
                layer: Module) \
        -> Tensor:
    """Small aux function to speed up reduce operation.
    :param layer_input: Input to the layer.
    :type layer_input: torch.Tensor
    :param layer: Layer.
    :type layer: torch.nn.Module
    :return: Output of the layer.
    :rtype: torch.Tensor
    """
    return layer(layer_input)


class DepthWiseSeparableConvBlock(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int], MutableSequence[int]],
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0,
                 dilation: Optional[int] = 1,
                 bias: Optional[bool] = True,
                 padding_mode: Optional[str] = 'zeros',
                 inner_kernel_size: Optional[Union[int,
                                                   Tuple[int, int], MutableSequence[int]]] = 1,
                 inner_stride: Optional[int] = 1,
                 inner_padding: Optional[int] = 0) \
            -> None:

        super(DepthWiseSeparableConvBlock, self).__init__()

        self.depth_wise_conv: Module = Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding),
            dilation=(1, dilation), groups=in_channels, bias=bias,
            padding_mode=padding_mode)

        self.non_linearity: Module = LeakyReLU()

        self.batch_norm: Module = BatchNorm2d(out_channels)

        self.point_wise: Module = Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(1, inner_kernel_size), stride=(1, inner_stride),
            padding=(0, inner_padding), dilation=1,
            groups=1, bias=bias, padding_mode=padding_mode)

        self.layers: List[Module] = [
            self.depth_wise_conv,
            self.non_linearity,
            self.batch_norm,
            self.point_wise]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.
        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return reduce(apply_layer, self.layers, x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth, kernel_num, grow=False):
        super(Block, self).__init__()
        assert out_channels % in_channels == 0

        self.p0 = DepthWiseSeparableConvBlock(in_channels, out_channels, kernel_size=kernel_num,
                                              stride=1)

        self.p01 = DepthWiseSeparableConvBlock(out_channels, out_channels, kernel_size=kernel_num,
                                               stride=stride)
        self.p1 = nn.ModuleList()

        for _ in range(depth):

            self.p1.append(DepthWiseSeparableConvBlock(out_channels, out_channels, kernel_size=kernel_num,
                                                       stride=1, padding=int((kernel_num-1)/2)))
        self.p1 = nn.Sequential(*self.p1)

    def forward(self, x):
        x = self.p0(x)
        x = self.p01(x)
        res = x
        x = self.p1(x)
        x = res + x

        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                
        n_embd,                
        n_head,                
        n_embd_ks,             
        max_len,               
        leads=2,
        arch=(2, 2, 8),      
        mha_win_size=[-1]*9,  
        scale_factor=2,     
        with_ln=True,       
        attn_pdrop=0.0,      
        proj_pdrop=0.0,      
        path_pdrop=0.1,      
        use_abs_pe=True,    
        use_rel_pe=False,    
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        self.n_in = n_in
        self.proj = None

        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                nn.Conv2d(
                    n_in, n_embd, (1, n_embd_ks),
                    stride=(1, 1), padding=(0, n_embd_ks//2), bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(nn.BatchNorm2d(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(
                self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        B, C, L, T = x.size()

        for idx in range(len(self.embd)):
            x = self.embd[idx](x)
            x = self.relu(self.embd_norm[idx](x))
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            x = x + pe[:, :, :T]

        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            x = x + pe[:, :, :T]
        for idx in range(len(self.stem)):
            x = self.stem[idx](x)
        out_feats = (x, )
        for idx in range(len(self.branch)):
            x = self.branch[idx](x)
            out_feats += (x, )
        return out_feats[-1]


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        B, C, T = x.size()
        assert T % self.stride == 0
        out_conv = self.conv(x)
        if self.stride > 1:
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            out_mask = mask.to(x.dtype)
        out_conv = out_conv
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        if self.affine:
            out *= self.weight
            out += self.bias
        return out

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class MaskedMHCA(nn.Module):

    def __init__(
        self,
        n_embd,         
        n_head,          
        n_qx_stride=1,  
        n_kv_stride=1,   
        attn_pdrop=0.0,  
        proj_pdrop=0.0,  
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.query_norm = nn.BatchNorm2d(self.n_embd)

        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.key_norm = nn.BatchNorm2d(self.n_embd)
        self.value_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.value_norm = nn.BatchNorm2d(self.n_embd)

        self.key = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv2d(self.n_embd, self.n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Conv2d(self.n_embd, self.n_embd, 1)

    def forward(self, x):
        B, C, L, T = x.size()

        q = self.query_conv(x)
        q = self.query_norm(q)

        k = self.key_conv(x)
        k = self.key_norm(k)
        v = self.value_conv(x)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        k = k.view(B, self.n_head, self.n_channels,
                   L, -1).transpose(2, 4).flatten(-2)
        q = q.view(B, self.n_head, self.n_channels,
                   L, -1).transpose(2, 4).flatten(-2)
        v = v.view(B, self.n_head, self.n_channels,
                   L, -1).transpose(2, 4).flatten(-2)

        att = (q * self.scale) @ k.transpose(-2, -1)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v
        out = out.transpose(2, 3).contiguous().view(B, C, L, -1)
        out = self.proj_drop(self.proj(out))
        return out

class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          
        n_head,         
        window_size,     
        n_qx_stride=1,   
        n_kv_stride=1,   
        attn_pdrop=0.0,  
        proj_pdrop=0.0,  
        use_rel_pe=False  
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.query_norm = nn.BatchNorm2d(self.n_embd)

        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.key_norm = nn.BatchNorm2d(self.n_embd)
        self.value_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.value_norm = nn.BatchNorm2d(self.n_embd)

        self.key = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv2d(self.n_embd, self.n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Conv2d(self.n_embd, self.n_embd, 1)

        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(
            affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:,
                                       :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -
                                    affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        x = x.view(total_num_heads, num_chunks, -1)
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1,
             window_overlap, window_overlap * 2 + 1)
        )

        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap:
        ]

        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        padded_value = nn.functional.pad(
            value, (0, 0, window_overlap, window_overlap), value=-1)

        chunked_value_size = (batch_size * num_heads,
                              chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(
            size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum(
            "bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x):
        B, C, L, T = x.size()

        q = self.query_conv(x)
        q = self.query_norm(q)

        k = self.key_conv(x)
        k = self.key_norm(k)
        v = self.value_conv(x)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(B, self.n_head, self.n_channels, L,
                   -1).transpose(2, 4).contiguous()
        k = k.view(B, self.n_head, self.n_channels, L,
                   -1).transpose(2, 4).contiguous()
        v = v.view(B, self.n_head, self.n_channels, L,
                   -1).transpose(2, 4).contiguous()
        q = q.view(B * self.n_head, -1, self.n_channels*L).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels*L).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels*L).contiguous()
        q *= self.scale
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)
        if self.use_rel_pe:
            att += self.rel_pe
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        out = out.transpose(2, 3).contiguous().view(B, C, L, -1)
        out = self.proj_drop(self.proj(out))
        return out

class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,               
        n_head,                
        n_ds_strides=(1, 1),   
        n_out=None,            
        n_hidden=None,         
        act_layer=nn.GELU,     
        attn_pdrop=0.0,        
        proj_pdrop=0.0,        
        path_pdrop=0.0,        
        mha_win_size=-1,       
        use_rel_pe=False       
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        self.ln1 = nn.BatchNorm2d(n_embd)
        self.ln2 = nn.BatchNorm2d(n_embd)
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool2d(
                (1, kernel_size), stride=(1, stride), padding=(0, padding))
        else:
            self.pool_skip = nn.Identity()
        if n_hidden is None:
            n_hidden = 4 * n_embd  
        if n_out is None:
            n_out = n_embd
        self.mlp = nn.Sequential(
            nn.Conv2d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv2d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )
        self.drop_path_attn = nn.Identity()
        self.drop_path_mlp = nn.Identity()
    def forward(self, x, pos_embd=None):
        # Ref: https://arxiv.org/pdf/2002.04745.pdf
        out = self.attn(self.ln1(x))
        out = self.pool_skip(x) + self.drop_path_attn(out)
        out = out + \
            self.drop_path_mlp(self.mlp(self.ln2(out)))
        if pos_embd is not None:
            out += pos_embd
        return out

# Ref: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )
    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale

# Ref: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    ) 
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """
    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
