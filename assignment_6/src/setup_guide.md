# Assignment 6 Setup and Execution Guide

## 1. Environment Installation

Run all commands from inside `assignment_6/src/`:

```powershell
cd assignment_6/src
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- PyTorch >= 2.0 (uses CUDA automatically if available via `--device auto`, else falls back to CPU)
- NumPy, Pandas, Matplotlib, scikit-learn

## 2. Project Architecture

The codebase is organized into two main pipeline scripts and focused modular helpers:

```text
assignment_6/src/
├── part_1.py            # Complete end-to-end pipeline for Part 1 (Handwritten Characters)
├── part_2.py            # Complete end-to-end pipeline for Part 2 (CV Speech Segments)
├── models.py            # Unified sequence classifier (SeqClassifier supporting RNN and LSTM)
├── data_loader.py       # Dataset discovery, parsing, sequence collation, and normalization
├── utils/
│   ├── convergence.py   # Absolute error convergence checker (|E_t - E_{t-1}| < 1e-4)
│   ├── metrics.py       # Evaluation, accuracy, loss, and confusion matrix calculation
│   ├── plotting.py      # Plotting curves, sample grids, heatmaps, and summary markdown
│   ├── io.py            # File I/O, JSON, CSV/Markdown table export, and logger setup
│   └── seed.py          # Deterministic seeding helper
├── data/                # Dataset storage (1_handwritten and 2_CV)
├── models/              # Saved winning checkpoints (*_best.pt)
└── results/             # Generated figures, tables, logs, and summary reports
```

## 3. Dataset Preprocessing & Visualizations

- **Part 1 — Handwritten Character Dataset (`1_handwritten`, 5 classes: `a`, `chA`, `dA`, `lA`, `tA`):**
  - Input coordinate sequences `(x, y)` are independently min-max normalized to `[0, 1]`.
  - Individual stroke visualizations featuring ultra-thin connecting lines and small red vertex markers are exported to `data/1_handwritten/normalized_images/<class>/<split>/<sample>.png`.
  - A 5x6 sample grid (`sample_grid.png`) is generated for qualitative visual assessment.
- **Part 2 — Consonant-Vowel Segment Dataset (`2_CV`, 5 classes: `bhA`, `hai`, `ka`, `ne`, `tA`):**
  - 39-dimensional acoustic MFCC frame sequences.
  - Feature standardization (Z-score mean and std) is computed strictly on the training split and applied to validation/test sets.

## 4. Experimental Protocol

- **Split Ratio:** Stratified 70% train / 15% validation / 15% test.
- **Stopping Rule:** Converged when average epoch change in training error satisfies $|E_t - E_{t-1}| < 10^{-4}$ with patience 5, or safety ceiling of 3000 epochs. Best models are strictly selected from runs that converged before hitting the max epoch count.
- **2-Stage Hyperparameter Study (15 configs per model type, 30 per dataset):**
  - **Stage 1 (9 configs):** Grid of hidden sizes `[32, 64, 128]` $\times$ layers `[1, 2, 3]`.
  - **Stage 2 (6 configs):** Tuning around the Stage 1 winner:
    - 2 alternate learning rates (`3e-3`, `3e-4`)
    - RMSprop optimizer
    - SGD optimizer with momentum 0.9 (`lr=1e-2`)
    - Dropout (`0.2`)
    - Bidirectional RNN / LSTM
- **Zero Retraining:** The winning model state is captured directly in-memory and saved to `models/<dataset>_<model>_best.pt`.
- **Total Models:** 30 models in Part 1 + 30 models in Part 2 = 60 total models.

## 5. Execution Commands

From `assignment_6/src/`:

```powershell
# Run the complete Part 1 pipeline (Handwritten Characters: data prep, RNN + LSTM sweeps, plots)
python part_1.py

# Run the complete Part 2 pipeline (CV Segments: data prep, RNN + LSTM sweeps, plots)
python part_2.py
```

### Useful CLI Flags

```powershell
# Run only LSTM or only RNN
python part_1.py --model lstm
python part_2.py --model rnn

# Fast smoke test / dry run (e.g., 5 epochs max)
python part_1.py --max_epochs 5

# Re-generate plots and summary tables without re-training
python part_1.py --plots_only
python part_2.py --plots_only
```
