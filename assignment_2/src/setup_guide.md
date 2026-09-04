# Setup Guide

All commands below must be run from inside the `Group30_Assignment2_code` folder.

## Folder Contents

The `Group30_Assignment2_code` folder contains:

- `fcnn.py`: Fully Connected Neural Network implemented from scratch (NumPy only).
- `model_selection.py`: Architecture grid generation, cross-validation sweep, and best-model selection.
- `classification.py`: Full classification pipeline for Linearly Separable (LS) and Non-Linearly Separable (NLS) datasets.
- `compare_with_a1.py`: Compares Assignment 2's best FCNN performance against Assignment 1's single-neuron model.
- `regression.py`: Stub for regression tasks (not yet implemented in this pass).
- `multiclass.py`: Carried over from Assignment 1 for reference and baselines.
- `perceptron.py`: Carried over from Assignment 1 for reference.
- `requirements.txt`: Dependencies specification.
- `data/`: Datasets for classification and regression.
- `utils/`: Data loading, activation functions, performance metrics, and visualization utilities.
- `results/`: Experiment artifacts, metrics, and plots.

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

## Run Classification Pipeline

Run the complete classification experiments:

```powershell
python classification.py
```

This script performs:
1. Stratified 60% train / 20% validation / 20% test splitting.
2. Architecture and hyperparameter sweeps over both LS (1 hidden layer) and NLS (2 hidden layers).
3. Evaluation metric logging (`evaluation_metrics.txt`) and decision region plotting (`decision_region.png`) for all candidate architectures under `results/<dataset>/sweep/<config_id>/`.
4. Selection of the best architecture based on validation metrics, followed by test set evaluation under `results/<dataset>/best/`.
5. Generation of error curves, decision regions, and 3D node output surface plots across all splits for the best models.
6. A summary table saved to `results/summary.txt`.

Results are saved under:
- `results/LS/sweep/`
- `results/LS/best/`
- `results/NLS/sweep/`
- `results/NLS/best/`

## Compare with Assignment 1

To compare the best FCNN architectures with Assignment 1's single-neuron results:

```powershell
python compare_with_a1.py
```

Comparison table will be printed to stdout and saved to:
- `results/a1_vs_a2_comparison.txt`

## Complete Run Sequence

```powershell
python -m pip install -r requirements.txt
python classification.py
python compare_with_a1.py
```
