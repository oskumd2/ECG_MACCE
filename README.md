# Multimodal Deep Learning for Postoperative MACCE Prediction

This repository contains the official codebase for the paper **"Multimodal Deep Learning to Predict Postoperative Major Adverse Cardiac and Cerebrovascular Events after Non-Cardiac Surgery"** published in the International Journal of Surgery, 2025.

## Overview

This project implements a multimodal deep learning approach to predict postoperative Major Adverse Cardiac and Cerebrovascular Events (MACCE) after non-cardiac surgery. The model combines 12-lead electrocardiogram (ECG) data with clinical features including age, gender, and surgical procedure information to achieve superior predictive performance compared to traditional risk assessment tools.

## Key Features

- **Multimodal Architecture**: Combines ECG signals with clinical metadata
- **Deep Learning Model**: Custom CNN-Transformer architecture for ECG processing
- **Clinical Integration**: Incorporates age, gender, and surgical procedure codes (ICD-10)
- **Comprehensive Evaluation**: Extensive validation with multiple metrics and subgroup analyses
- **Interpretability**: SHAP analysis and saliency mapping for model explainability

## Model Architecture

### ECG Processing Pipeline
- **12-lead ECG input**: Standard 12-lead ECG recordings (5000 samples per lead)
- **CNN-Transformer backbone**: Custom architecture combining convolutional blocks with transformer attention
- **Feature extraction**: Learned representations from ECG signals
- **Multimodal fusion**: Integration with clinical features (age, gender, surgical codes)

### Clinical Features
- **Demographics**: Age and gender
- **Surgical procedures**: ICD-10 codes for procedure (P), operation (O), and approach (A)
- **Risk scores**: ASA (American Society of Anesthesiologists) physical status classification
- **Laboratory values**: Troponin I levels (when available)

## Repository Structure

```
├── main.ipynb                 # Main analysis notebook
├── model_code.py              # Neural network model definitions
├── model/
│   └── blocks.py             # CNN-Transformer architecture
├── team_code.py              # Data loading and preprocessing
├── helper_code.py            # Utility functions
├── plot_model.py             # Visualization and evaluation functions
├── evaluate_model.py         # Model evaluation metrics
├── delong.py                 # Statistical testing (DeLong's test)
└── model/                    # Trained model weights
    └── auc=0.8772_.../      # Best performing model
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install scikit-learn scikit-learn-intelex
pip install pandas numpy matplotlib seaborn
pip install xgboost shap optuna
pip install neurokit2 joblib h5py
pip install tensorflow keras
pip install wfdb tqdm
```

### Data Setup

1. Place ECG data files in the appropriate directory structure
2. Ensure clinical metadata CSV files are available
3. Update file paths in the configuration

## Key Components

### 1. Data Preprocessing (`team_code.py`)
- ECG signal loading and preprocessing
- Clinical metadata integration
- Data augmentation and normalization

### 2. Model Architecture (`model/blocks.py`)
- **FinalModel**: Main multimodal architecture
- **ConvTransformerBackbone**: CNN-Transformer hybrid
- **Block**: Residual convolutional blocks

### 3. Training Pipeline (`main.ipynb`)
- Hyperparameter optimization with Optuna
- Cross-validation with GroupKFold
- Early stopping and model selection

### 4. Evaluation Framework (`plot_model.py`, `delong.py`)
- Comprehensive metrics (AUROC, AUPRC, F1, etc.)
- Statistical significance testing
- Calibration analysis
- Subgroup performance analysis

## Clinical Applications

This model can be used for:

1. **Preoperative risk assessment** for non-cardiac surgery patients
2. **Clinical decision support** for perioperative management
3. **Resource allocation** based on predicted MACCE risk
4. **Patient counseling** regarding surgical risks

## Model Interpretability

The repository includes tools for model interpretability:

- **SHAP analysis**: Feature importance for clinical features
- **Saliency mapping**: ECG signal regions contributing to predictions
- **Subgroup analysis**: Performance across different patient populations
- **Decision curve analysis**: Clinical utility assessment

## Citation

If you use this code in your research, please cite:

```bibtex
@article{multimodal_macce_2025,
  title={Multimodal Deep Learning to Predict Postoperative Major Adverse Cardiac and Cerebrovascular Events after Non-Cardiac Surgery},
  author={[Authors]},
  journal={International Journal of Surgery},
  year={2025}
}
```

## Contact

For questions or issues, please contact oskumd00@gmail.com.

## Acknowledgments

- PhysioNet Challenge framework
- Seoul National University Hospital for data
- Computing resources and collaborators

---

**Note**: This is the official implementation of the multimodal deep learning approach for postoperative MACCE prediction. The model achieves state-of-the-art performance by combining ECG signals with clinical features in a unified deep learning framework. 
