# Applied Machine Learning in Perturbation Data

## Project Overview
Applied Machine Learning in Genomic Data Science Project: Single-gene Perturbations Classification based on Gene Expression Profiles

---

## Environment Setup

### 1. Create a Conda Environment

```bash
# Create a Conda environment named 'amlg_env' with Python 3.11.9
conda create --name amlg_env python=3.11.9

# Activate the Conda environment
conda activate amlg_env
```

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

---

## Install Dependencies

### 1. Install required libraries
```bash
pip install -r requirements.txt
```

---

## Running the Code

### 1. Execute the pipeline
```bash
python run_pipeline.py
```

### 2. Output
- Best hyperparameters selected using Hyperparameter tuning (grid search).
- Accuracy, classification report, confusion matrix, and AUC-ROC curve are saved in the `logs` directory.

---

## License
This project is licensed under the MIT License. See LICENSE file for details.

---
