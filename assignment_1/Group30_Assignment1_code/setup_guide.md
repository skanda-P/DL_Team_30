# Setup Guide

All commands below must be run from inside the 'src' folder.

## Folder Contents

The `src` folder should contain:

- `classification.py`
- `regression.py`
- `compare_results.py`
- `multiclass.py`
- `perceptron.py`
- `requirements.txt`
- `data/`
- `utils/`
- `results/`

The `data/` folder is required because the classification and regression scripts load their input files using paths relative to `src`.

## Prerequisites

Install Python 3.10 or later and confirm that Python is available:

```powershell
python --version
```

## Open the Project Folder

In PowerShell, change to the downloaded `src` folder. Replace the path with the location where you saved it:

```powershell
cd "D:\path\to\src"
```

Verify that the required files are present:

```powershell
Get-ChildItem
```

## Install Dependencies

Install the packages listed in `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Classification

Run the classification experiments from the `src` folder:

```powershell
python classification.py
```

This trains one-vs-one classifiers for both the linearly separable and non-linearly separable datasets using logistic and tanh activations. Evaluation files are written under:

```text
results/LS/
results/NLS/
```

## Run Regression

Run the regression experiments:

```powershell
python regression.py
```

This trains models for the univariate and bivariate datasets. Evaluation files are written under:

```text
results/Univariate/
results/Bivariate/
```

## Generate the Comparison Report

Run both experiment scripts before generating the report. Then run:

```powershell
python compare_results.py
```

The combined report is saved as:

```text
results_comparison_report.txt
```

## Run Everything in Order

The complete workflow is:

```powershell
python -m pip install -r requirements.txt
python classification.py
python regression.py
python compare_results.py
```
