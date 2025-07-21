# Longitudinal Brain Age Estimation

This repository provides a comprehensive pipeline for longitudinally robust brain age prediction using deep learning. The project includes data preprocessing, model training, inference, and bias correction, and is built using PyTorch Lightning.

## Features

- **Flexible Model Architectures:** Supports SFCN, DenseNet, ColesNet, and custom architectures.
- **Longitudinal Consistency:** Implements consistency loss functions and sampling strategies for longitudinal data.
- **Data Preprocessing:** Includes normalization, skull-stripping, and augmentation utilities.
- **Experiment Management:** Automated experiment directory creation, logging, and checkpointing.
- **Inference & Evaluation:** Scripts for inference, with metrics and plotting.
- **Bias Correction:** Post-hoc correction of predicted ages using robust regression (Coles et al. method).

## Project Structure

```
longitudinal_brain_age/
│
├── bias_correction/
│   └── coles_bias_correction.py        # Bias correction script (Cole et al.)
├── data/
│   └── dataset.csv                     # Dataset csv reference
├── inference/
│   └── config_inference.py             # Configuration file for inference
│   └── inference_main.py               # Main inference script
├── config.py                           # Configuration file for training 
├── data_module.py                      # Data loading, batching, and sampling
├── model_module.py                     # Model architectures and LightningModules
├── README.md                           # Project documentation
├── requirements.txt                    # Library requirements
├── train.py                            # Main training script
└── utils.py                            # Utility functions for preprocessing, transforms, etc.
```

## Getting Started

### 1. Environment Setup

- Python 3.8+
- Recommended: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation

- Prepare your CSV files with paths to MRI images and metadata (see example in `data/dataset.csv`).
- Update paths and parameters in `config.py` and `config_inference.py` as needed.
- Update paths in `bias_correction/coles_bias_correction.py`

### 3. Training

Run the main training script:

```bash
python train.py
```

- Supports both cross-sectional and longitudinal training (set in config).

### 4. Inference

Run inference on validation, test, or external datasets:

```bash
python inference/inference_main.py
```

- Results and metrics are saved to the experiment folder.
- Plots of predicted vs. real ages are generated.

### 5. Bias Correction

Apply post-hoc bias correction to predicted ages:

```bash
python bias_correction/coles_bias_correction.py
```

- Uses Huber regression to correct for age bias (Coles et al.).
- Saves corrected predictions and updated metrics.

## Key Modules

- **model_module.py:** Defines model architectures and training logic.
- **data_module.py:** Handles data loading, batching, and custom sampling for longitudinal pairs.
- **utils.py:** Preprocessing, normalization, augmentation, and helper functions.
- **train.py:** Orchestrates training, logging, and checkpointing.
- **inference/inference_main.py:** Loads trained models and runs inference.
- **bias_correction/coles_bias_correction.py:** Applies bias correction to predictions.

## Customization

- Modify `config.py` and `config_inference.py` to adjust model, data, and training parameters.
- Add new architectures or loss functions in `model_module.py`.
- Extend data augmentation or preprocessing in `utils.py`.

## Citation

If you use this codebase in your research, please cite the article 'Title'.

## License

This project is released under the MIT License.

---

**Contact:**  
For questions or contributions, please open an issue or contact me at clara.lisazo@udg.edu.
