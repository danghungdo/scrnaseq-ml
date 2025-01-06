# Applied Machine Learning in Perturbation Data

## Project Overview
Applied Machine Learning in Genomic Data Science Project: A machine learning approach for classification based on the Norman perturbation dataset. 

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
- Best hyperparameters selected using Hyperparameter tuning.
- Accuracy, classification report, and confusion matrix printed for evaluation.

---

## Notes
- Replace the dummy dataset in the script with real data if available.
- Adjust hyperparameters in the `param_grid` section of the script for further optimization.
- Ensure Conda is installed and configured correctly.

---

## License
This project is licensed under the MIT License. See LICENSE file for details.

---
