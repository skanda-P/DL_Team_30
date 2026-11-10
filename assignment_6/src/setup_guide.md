# Assignment 6 Setup and Execution Guide (Group 30)

## 1. Environment Installation

Run all commands from inside `assignment_6/src/`:

```powershell
cd assignment_6/src
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- PyTorch >= 2.0 (uses CUDA automatically via `--device auto`, or falls back to CPU)
- NumPy, Pandas, Matplotlib, scikit-learn

## 2. Dataset Layout & Normalized Handwritten Images

- **Dataset 1 (Handwritten Character Strokes — classes `a`, `chA`, `dA`, `lA`, `tA`):**
  - `src/data/1_handwritten/<class>/train/*.txt` (345 training files) and `dev/*.txt` (100 test files).
  - Each file's `(x, y)` coordinates are independently min-max normalized to `[0, 1]`.
  - All 445 normalized stroke images are exported separately inside `src/data/1_handwritten/normalized_images/<class>/<split>/<sample_stem>.png`.
- **Dataset 2 (Consonant-Vowel Speech MFCCs — classes `bhA`, `hai`, `ka`, `ne`, `tA`):**
  - `src/data/2_CV/<class>/Train/*.mfcc` (2,460 training files) and `Test/*.mfcc` (614 test files), 39-D MFCC frames.
  - Standardized strictly using training-split mean and standard deviation.

## 3. Splits & Stopping Rule

- Fixed stratified 80/20 train/val split (`--split_seed 1234`), persisted to `results/<dataset>/split.json`. Test (`dev`/`Test`) data is evaluated strictly at the end for reporting.
- Fixed single training seed (`--seed 42`) with `CrossEntropyLoss` and convergence stopping when `|E_t - E_{t-1}| < 1e-4` (`max_epochs=1000` safety cap).

## 4. Exact Reproduction Commands

Execute from inside `assignment_6/src/`:

```powershell
# 1. Verify dataset discovery, padding invariance, and export all normalized handwritten images
python data_loader.py --dataset handwritten --selftest --export_all_images
python data_loader.py --dataset cv --selftest

# 2. Run 2-stage hyperparameter sweep (15 configs per model, 1 run each, zero retraining)
python run_experiments.py --dataset handwritten
python run_experiments.py --dataset cv

# 3. Generate all report plots, confusion matrices, normalized grids, and results/summary.md
python generate_report_plots.py
```

*(Single-configuration runs can also be launched via `python train.py --dataset handwritten --model lstm --hidden_size 64 --num_layers 2 --lr 1e-3`.)*
