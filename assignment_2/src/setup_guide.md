# Setup Guide

All commands below must be run from inside the `Group30_Assignment2_code` folder.

## Folder Contents

The `Group30_Assignment2_code` folder contains:

- `fcnn.py`: Fully Connected Neural Network implemented from scratch (NumPy only) with early stopping convergence criteria (patience=5, loss threshold=0.0001).
- `model_selection.py`: Architecture grid generation, cross-validation sweep, and best-model selection.
- `classification.py`: Full classification pipeline for Linearly Separable (LS) and Non-Linearly Separable (NLS) datasets.
- `regression.py`: Full regression pipeline for Univariate and Bivariate datasets.
- `classification_a1.py`: Assignment 1 single-neuron classification baseline trained with convergence criteria on the exact same 60/20/20 split.
- `regression_a1.py`: Assignment 1 single-neuron linear regression baseline trained with convergence criteria on the exact same 60/20/20 split.
- `compare_results.py`: Aggregates all hyperparameter sweeps into structured summary reports for both A2 (`results/summary.txt`) and A1 (`results/a1/summary.txt`).
- `compare_with_a1.py`: Generates the head-to-head performance and convergence comparison report (`results/a1_vs_a2_comparison.txt`).
- `multiclass.py`: One-vs-One multi-class classifier using single-neuron perceptron with early stopping.
- `perceptron.py`: Single-neuron perceptron with early stopping.
- `requirements.txt`: Dependencies specification.
- `data/`: Datasets for classification and regression.
- `utils/`: Data loading, activation functions, performance metrics, and visualization utilities.
- `results/`: Experiment artifacts, metrics, and plots for all A2 and A1 models.

## Prerequisites

Python 3.10 or later is recommended. Confirm Python availability:

```powershell
python --version
```

## Install Dependencies

Install the required packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

## Run Pipelines

### 1. Classification (Assignment 2 - FCNN)
```powershell
python classification.py
```
Performs 60/20/20 split, sweeps architectures and learning rates with early stopping (patience=5, diff=0.0001), evaluates test performance, and saves decision regions, error curves, and 3D node output surfaces under `results/LS/` and `results/NLS/`.

### 2. Regression (Assignment 2 - FCNN)
```powershell
python regression.py
```
Evaluates Univariate (1 hidden layer) and Bivariate (1 & 2 hidden layers) regression models, logs RMSE & %RMSE on train, val, and test splits, and generates fit, scatter, error, and 3D node output plots under `results/Univariate/` and `results/Bivariate/`.

### 3. Assignment 1 Baseline Pipelines
```powershell
python classification_a1.py
python regression_a1.py
```
Trains single-neuron models on the exact same 60/20 train/validation splits and evaluates on the exact same 20% test split with the identical convergence rule. Saves results directly to `results/a1/`.

### 4. Generate Summaries & Comparison Reports
```powershell
python compare_results.py
python compare_with_a1.py
```
- `compare_results.py`: Generates `results/summary.txt` (A2) and `results/a1/summary.txt` (A1).
- `compare_with_a1.py`: Generates `results/a1_vs_a2_comparison.txt` containing side-by-side performance tables (accuracy, RMSE, %RMSE, convergence epochs, speedup factors, and inferences).

## Complete Execution Sequence

```powershell
python -m pip install -r requirements.txt
python classification_a1.py
python regression_a1.py
python classification.py
python regression.py
python compare_results.py
python compare_with_a1.py
```
