# 🔍 Telco Customer Churn Prediction

This project aims to predict customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It covers the full ML workflow: from data preprocessing and EDA to model training, evaluation, and optimization.

---

## 🚀 Tech Stack

- **Python 3.7+**
- **PyTorch**: For model building and training
- **Optuna**: For hyperparameter tuning
- **FastAPI**: For serving the model as an API
- **Uvicorn**: ASGI server to run FastAPI
- **AWS EC2**: Model deployed on a Linux-based virtual machine
- **SCP / SSH**: To copy files and manage the remote server
- **Boto3**: (optional) for AWS interactions from Python

> ✅ Model was deployed on **Amazon Web Services (AWS)** using **EC2 instances** to explore, understand, and practice real-world ML deployment workflows.

---

## 📁 Project Structure

```
.
├── config/  
│   └── config.yaml                  # YAML file for training configuration  
├── data/  
│   ├── raw/                         # Contains raw dataset (CSV)  
│   └── processed/                   # Preprocessed data files  
├── models/  
│   └── logistic_regression_v1/     # Saved model and config  
├── notebooks/  
│   └── 01_eda.ipynb                 # Exploratory Data Analysis notebook  
├── reports/                         # Placeholder for future reports/figures  
├── src/  
│   ├── data_prep.py                 # Data cleaning and encoding  
│   ├── evaluate.py                  # Model evaluation metrics  
│   ├── model.py                     # PyTorch logistic regression model  
│   ├── optimize.py                  # Hyperparameter tuning using Optuna  
│   ├── train.py                     # Model training pipeline  
│   └── utils.py                     # Utility functions (if extended)  
├── main.py                          # Main training + evaluation script  
├── test_api.py                      # Script to send a sample request to the deployed model  
├── requirements.txt  
└── README.md
```

---

## 📊 EDA (Exploratory Data Analysis)

EDA is conducted in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). It includes:

- Feature distributions
- Churn vs non-churn behavior
- Correlation & trends
- Initial logistic regression analysis

---

## ⚙️ Configuration

All parameters (hyperparams, data paths, thresholds, device settings) are defined in `config/config.yaml`.

Example:
```yaml
model:
  lr: 0.0009
  weight_decay: 0.0015
  batch_size: 32
  max_epochs: 50
  threshold: 0.5751
  device: "cpu"
  seed: 42

data:
  path: "data/processed/telco_churn_processed.csv"
```

---

## 🛠️ How to Run

### 1. 📦 Preprocess Data
```bash
python src/data_prep.py
```

### 2. 🧠 Train & Evaluate Model
```bash
python main.py
```

### 3. 🔍 Hyperparameter Tuning (Optional)
```bash
python -c "from src.optimize import run_optimization; run_optimization()"
```

---

## 📈 Evaluation Metrics

Implemented via `src/evaluate.py`:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (probability-based)
- Confusion Matrix

---

## 💾 Model

Model is a simple logistic regression built with PyTorch (`src/model.py`), trained with early stopping based on validation AUC.

---

## 🔐 Reproducibility

- Configurable via `config.yaml`
- Fixed random seed
- Trained with `train_test_split(stratify=True)`

---

## 📋 Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```
