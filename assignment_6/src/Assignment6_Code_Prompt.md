# Prompt: CS601T Assignment 6 — RNN & LSTM Sequence Classification (Code Only)

## 0. Role and scope

You are an expert deep-learning engineer. Write the **complete, runnable Python code** for CS601T (Deep Learning) Programming Assignment 6, for **Group 30**.

- Deliver **code only**. Do **not** write the report. The code must, however, produce every table, plot, and log the report needs (see Section 7).
- Use **PyTorch** (allowed by the assignment). Python 3.10+. All code in `.py` files (no notebooks).
- Priorities: **correctness, reproducibility, rigorous experimentation, simple structure**. No over-engineering: no config-framework scaffolding, no experiment-tracking servers.
- The data is **already placed** in the project. Do not download or generate data. Do not hardcode machine-specific absolute paths.

---

## 1. Assignment specification (authoritative)

**Goal:** Build an **RNN** and an **LSTM** classifier for each of two datasets, tune hyperparameters to find the best RNN and best LSTM per dataset, and report results.

### Dataset 1 — Handwritten character dataset (Kannada/Telugu)
- 5-class data (the classes assigned to Group 30 are listed in `Mapping_Handwritten.txt`).
- Each file = **one sample** = one pen stroke (pen-down to pen-up) of one character.
- File format: a flat array of numbers.
  - **1st element** = number of 2-D points `N` in that file.
  - **2nd element onwards** = the points, read **in pairs**: `(x1, y1), (x2, y2), ...` (so elements 2–3 are point 1, elements 4–5 are point 2, ...).
- **Normalization (required):** in each file, x and y coordinates must be normalized to the range **[0, 1]**.
- Training and test data are separated and provided inside the respective character folders; **test data is in the `dev` folder**.
- Input feature dimension: 2 (x, y). Sequences have variable length.

### Dataset 2 — Consonant-Vowel (CV) segment dataset (Hindi speech)
- 5-class data (classes listed in `Mapping_CV.txt`).
- Each file = **one sample** = one utterance of a CV segment.
- Each **row** in a file is one **39-dimensional MFCC** frame vector; the number of rows (time steps) varies per file.
- Training and test data are separated inside the respective CV-segment folders.

### Modeling and training requirements
1. Build a **vanilla RNN** and an **LSTM** for **each** dataset (4 model families total: `{handwritten, cv} x {rnn, lstm}`).
2. Obtain the **best RNN and best LSTM per dataset** by tuning hyperparameters, including at least: **number of recurrent (hidden) layers** and **number of neurons (hidden units) per layer at a time step**. More hyperparameters are encouraged (Section 5).
3. **Convergence criterion (strict):** stop training when the **absolute difference between the average training errors of two successive epochs falls below 1e-4**, i.e. `|E_t - E_{t-1}| < 1e-4`. **Do not use a fixed number of epochs as the convergence criterion.** (Only a very large safety cap is allowed; see Section 4.3.)
4. Required outputs per the assignment:
   - (a) Plot of **average training error (y) vs. epochs (x)** for **each hyperparameter set**.
   - (b) **Training accuracy and test accuracy for each hyperparameter set**.
   - (c) **Training accuracy, test accuracy, and confusion matrix** for the **best** hyperparameter set.
   - (d) **At least 5 images (after normalization) from each of the 5 handwritten character classes**.
5. Each group must use **only its own** assigned dataset classes (the 5 class folders present in the data directory).

### Submission constraints the code must respect
- Code folder name: `Group30_Assignment6_code`; zipped as `Group30_Assignment6_code.zip`.
- Code must be in `.py` files.

---

## 2. Project layout (keep it simple and flat; everything lives under `src/`)

Run everything **from inside `src/`**. Data, trained models, and results are nested inside `src/`.

```
Group30_Assignment6_code/
├── requirements.txt
├── setup_guide.md            # how to install, where data goes, exact commands to reproduce every result
├── contributions.md          # template for team contributions (leave names blank for me to fill)
└── src/
    ├── data_loader.py        # discovery, parsing, normalization, splits, Dataset/DataLoader, padding collate
    ├── models.py             # RNN/LSTM classifier (single configurable class)
    ├── train.py              # train ONE configuration (CLI) with the convergence criterion
    ├── run_experiments.py    # full staged hyperparameter study for one dataset (CLI)
    ├── final_eval.py         # best-config selection, multi-seed retrain, test eval, confusion matrix
    ├── generate_report_plots.py  # all figures/tables for the report from saved results
    ├── utils/
    │   ├── seed.py           # set_seed, deterministic flags
    │   ├── convergence.py    # ConvergenceChecker
    │   ├── metrics.py        # accuracy, confusion matrix, per-class P/R/F1
    │   └── io.py             # JSON/CSV helpers, run-id hashing, logging setup
    ├── models/               # saved checkpoints (.pt)
    ├── results/              # per-dataset/per-model run histories, CSVs, plots, summary.md
    └── data/                 # (already placed by me; do not modify)
```

Do not add extra top-level folders, config systems, or class hierarchies beyond this.

---

## 3. Data pipeline (`data_loader.py`)

### 3.1 Robust discovery (do not assume the exact folder nesting)
I have placed the data correctly, but the exact nesting may be either `<class>/train` + `<class>/dev` or `train/<class>` + `dev/<class>`-style. Implement a **discovery routine** that:
- Takes `DATA_ROOT` (a constant at the top of `data_loader.py` plus a `--data_root` CLI override) and `dataset ∈ {handwritten, cv}`.
- Walks the tree and identifies the 5 class folders and, for each class, its **training files** and **test (dev) files**. Support both nesting orders above.
- **Prints a discovery summary** (class names, number of train files, number of test files per class) and **fails loudly with a clear error** if it does not find exactly 5 classes or finds zero train/test files.
- Uses the **class folder names as labels** everywhere (including plots). **Do not render Kannada/Telugu glyphs** in matplotlib (font issues); if `Mapping_*.txt` is present, parse it best-effort only to print a mapping in the log.
- Encodes labels as integers `0..4` sorted by class folder name (deterministic).

### 3.2 Parsing
- **Handwritten:** read all numbers (whitespace/comma/newline-separated; be tolerant). `N = int(first)`; points = remaining numbers reshaped to `(-1, 2)`. **Validate** that the number of remaining values is `2N`; if mismatched, truncate/trim to the available complete pairs, **log a warning counter** (do not crash), and report the count of malformed files at the end of loading.
- **CV:** read a 2-D array with 39 columns per file (rows = frames, whitespace- or comma-separated). Validate 39 columns; warn on mismatch.
- Skip hidden files and non-data files (`.DS_Store`, etc.).

### 3.3 Preprocessing
- **Handwritten (as specified):** per file, min-max normalize **x and y independently to [0, 1]** (guard against zero range with an epsilon). This is the default and is **required** by the assignment.
- **CV:** standardize each of the 39 features with **mean/std computed on the training split only** (never on val/test). Store the statistics so the exact same transform is applied to val/test.
- Provide an optional, **off-by-default** feature switch `--add_deltas` that appends first-difference features (handwritten: `dx, dy` -> 4-dim input; CV: frame-to-frame delta -> 78-dim). Used only in an ablation (Section 5).

### 3.4 Splits (critical for honest experimentation)
- **Test set = the provided `dev` data. It must never influence any training or model-selection decision.**
- Create a **stratified validation split from the training data** (default 20%) using a **fixed `split_seed` (e.g., 1234) that is independent of the model seed**, so **every configuration sees exactly the same train/val partition**. Persist the split indices to `results/<dataset>/split.json` and reload them on later runs.
- The model/run seed (`--seed`) controls only weight initialization and shuffling.

### 3.5 Batching variable-length sequences
- `collate_fn` pads sequences within a batch, returns `(padded_x, lengths, labels)`.
- In the model, use `pack_padded_sequence(..., enforce_sorted=False)` so padding **never affects** the hidden state used for classification.
- Include a quick **sanity test** (a function or `--selftest` flag) that verifies a padded sequence and its unpadded version produce identical logits.

---

## 4. Model and training (`models.py`, `train.py`)

### 4.1 Model (`models.py`)
One configurable class `SeqClassifier` with arguments:
`rnn_type ∈ {"rnn","lstm"}`, `input_size`, `hidden_size`, `num_layers`, `bidirectional`, `dropout` (inter-layer, applied only when `num_layers > 1`; if `num_layers == 1` apply it on the pooled representation instead), `head_hidden` (0 = single linear layer to 5 classes; >0 = one extra `Linear -> ReLU` layer of that width before the output), `pooling ∈ {"last","mean"}`.
- `"last"` = final hidden state of the **last layer at the true last time step** (for bidirectional: concatenate forward-final and backward-final).
- `"mean"` = masked mean over valid time steps of the top layer's outputs.
- Vanilla RNN uses `nn.RNN` with `tanh` nonlinearity. Do not silently swap in GRU.
- Expose `count_parameters()`.

### 4.2 Training loop (`train.py`)
- Loss: `CrossEntropyLoss`. Optimizer configurable: `adam` (default), `rmsprop`, `sgd` (with momentum 0.9). Configurable `lr`, `batch_size`, `grad_clip` (max-norm; 0/None disables).
- Shuffle training data each epoch (seeded).
- **Average training error per epoch** `E_t` = mean cross-entropy over **all training samples**, computed with a **clean evaluation pass at the end of every epoch** (`model.eval()`, `torch.no_grad()`, no dropout) so the curve is deterministic and comparable across configs. (Offer `--error_mode running` as an alternative that averages the training-pass batch losses, weighted by batch size; default is `eval`.)
- Also record, per epoch: train accuracy, **val loss and val accuracy**, epoch wall-time. (Val metrics are for diagnostics and model selection only; they must **not** be used as a stopping criterion.)

### 4.3 Convergence (`utils/convergence.py`)
- `ConvergenceChecker(tol=1e-4, min_epochs=2, max_epochs=1000)`.
- Stop when `|E_t - E_{t-1}| < 1e-4` (and `t >= min_epochs`).
- `max_epochs` is **only a safety cap** to prevent infinite loops; it must be large (default 1000) and **every run must log its `stop_reason ∈ {"converged","max_epochs_cap"}`**. Runs that hit the cap are flagged `converged=False` in all result tables and marked in plots.
- **No early stopping on validation loss, no fixed-epoch schedules, no LR schedulers that change the stopping behavior.**
- Unit-test the checker on a synthetic error sequence.

### 4.4 Single-run CLI (`train.py`)
Example: `python train.py --dataset handwritten --model lstm --hidden_size 64 --num_layers 2 --lr 1e-3 --batch_size 32 --seed 0`
Each run saves, under `results/<dataset>/<model>/<stage>/<run_id>/`:
- `config.json` (all hyperparameters + seed + split_seed + library versions),
- `history.json` (per-epoch `train_error`, `train_acc`, `val_loss`, `val_acc`, time),
- `metrics.json` (final train/val/test accuracy, epochs to converge, stop_reason, param count, total train time),
- the final model weights to `models/` **only for runs flagged as "keep"** (final/best runs), to avoid thousands of checkpoints.

Test accuracy is computed **once at the end of each run for reporting (assignment item b)** and is **never used for selection**.

---

## 5. Experimental protocol (`run_experiments.py`) — this is the core; do it properly

Run for `--dataset {handwritten,cv}` and both `--model {rnn,lstm}` (loop over both unless specified). Use **`--seeds 0 1 2` by default** (configurable) so every configuration is run with multiple seeds and reported as **mean ± std**. Use a **fixed baseline** and change **one factor group at a time**; log which stage each run belongs to.

**Baseline config:** `hidden_size=64, num_layers=1, bidirectional=False, dropout=0, head_hidden=0, pooling="last", optimizer=adam, lr=1e-3, batch_size=32, grad_clip=1.0, no delta features`.

### Stage 1 — Architecture grid (the assignment's required hyperparameters)
Full grid: `hidden_size ∈ {16, 32, 64, 128}` x `num_layers ∈ {1, 2, 3}` (12 configs per model, per dataset), baseline otherwise.
Select the best architecture by **mean validation accuracy over seeds** (tie-break 1: lower mean val loss; tie-break 2: fewer parameters).

### Stage 2 — Optimization hyperparameters (at the Stage-1 best architecture)
- `lr ∈ {1e-2, 3e-3, 1e-3, 3e-4}` x `batch_size ∈ {16, 32, 64}` (full grid, 12 configs).
- `optimizer ∈ {adam, rmsprop, sgd}` (at the best lr/batch from the grid above; for `sgd` also try lr 1e-1 and 1e-2 since it needs larger rates).
- `grad_clip ∈ {off, 1.0}` (important to expose vanilla-RNN instability).

### Stage 3 — Architectural ablations (at the best config so far)
One factor at a time:
- `bidirectional ∈ {False, True}`
- `dropout ∈ {0.0, 0.2, 0.4}`
- `pooling ∈ {last, mean}`
- `head_hidden ∈ {0, 64}`
- input features: base vs `--add_deltas`
- CV only: with vs without train-set standardization (to quantify its effect)

### Stage 4 — Final model per (dataset, model)
- Take the **overall best configuration** (by mean val accuracy across all stages, same tie-breaks).
- Re-train it with **5 seeds** (`0..4`), report mean ± std of train/val/test accuracy and epochs to converge.
- The **"best model" for the confusion matrix** (assignment item c) = the seed with the highest **validation** accuracy among those 5; save its checkpoint to `models/`. Also save the **sum of confusion matrices across all 5 seeds** as a robustness view.

### Additional required analyses (cheap but valuable for the report's observations)
- **RNN vs LSTM comparison** per dataset: best-config table side by side, plus paired comparison across matched configs from Stage 1 (e.g., a table of LSTM-minus-RNN val/test accuracy per (hidden, layers)).
- **Effect of depth and width** summarized as heatmaps (hidden_size x num_layers -> mean val accuracy) for each (dataset, model).
- **Convergence behavior:** epochs-to-converge vs configuration; fraction of runs that hit the safety cap.
- **Sequence-length analysis (best model):** test accuracy bucketed by sequence-length quartile.
- **Per-class precision/recall/F1** on test for the best model and the most-confused class pairs.
- **Class balance and sequence-length distributions** per dataset (train/test).

### Efficiency and robustness
- Skip runs whose `metrics.json` already exists (resume support) unless `--force`.
- Use GPU if available, CPU otherwise; keep the code runnable on CPU in reasonable time (small models; batch sizes as specified).
- Print a compact progress line per run (config, epochs, train/val/test acc, seconds).
- Wrap each run so one failure (e.g., NaN loss) is logged with its config and does not abort the sweep; NaN runs are recorded as `diverged=True`.

---

## 6. Reproducibility and rigor checklist (must be enforced in code)

- Seed `random`, `numpy`, `torch` (CPU + CUDA); set deterministic cuDNN flags where possible; document any remaining nondeterminism in `setup_guide.md`.
- Fixed `split_seed` for the train/val split; persisted split indices.
- Normalization statistics (CV standardization) fit on **train only**.
- **Test data touched only to compute reported test metrics**, never for tuning, early stopping, or selection. Add an assertion/comment at the selection code making this explicit.
- Every run's full config and library versions saved.
- All tables saved both as **CSV** and as **Markdown**.
- No hardcoded dataset-specific numbers (class counts, sequence lengths) outside discovery.

---

## 7. Outputs the report will need (`generate_report_plots.py`)

Everything is generated from saved results (no retraining). Save under `results/<dataset>/...` as PNG (dpi >= 200) and CSV/MD:

**Assignment item (a) — training error vs. epochs for every hyperparameter set**
- One PNG **per configuration** (x: epochs, y: average training error): all seeds as thin lines plus the seed-mean as a thick line; mark non-converged runs.
- Per-stage **overview figures** overlaying configurations (grouped sensibly, e.g., by `num_layers` with different `hidden_size` colors) so the report has compact figures.
- Use a **log-scale y option** for runs where the error spans orders of magnitude.

**Assignment item (b) — accuracy for each hyperparameter set**
- `results/<dataset>/<model>/all_configs.csv` and `.md`: one row per configuration with `stage, hidden_size, num_layers, bidirectional, dropout, head_hidden, pooling, optimizer, lr, batch_size, grad_clip, features, n_params, epochs_mean, converged_frac, train_acc_mean±std, val_acc_mean±std, test_acc_mean±std`.

**Assignment item (c) — best model**
- `best_model_summary.md/.csv`: best config, train/val/test accuracy (mean ± std over 5 seeds, and for the single best-val seed).
- **Confusion matrix** (counts and row-normalized) for the best model on **test**; also on **train**; saved as PNG with class-folder names as labels. Plus the seed-summed test confusion matrix.
- `classification_report.txt` (per-class precision/recall/F1).

**Assignment item (d) — handwritten samples**
- A grid PNG: **5 classes (rows) x at least 5 samples (columns, use 6)** of **normalized** strokes (x,y in [0,1]), drawn as a connected polyline with a start-point marker. Axis orientation handled by a constant `INVERT_Y` at the top of the plotting code (default `False`; I will check the orientation visually and flip if the characters look upside down). Also save one PNG per class.

**Extra analysis figures:** depth/width heatmaps, RNN-vs-LSTM bar chart with error bars, epochs-to-converge plot, sequence-length-bucket accuracy, class balance and length distribution plots.

**`results/summary.md`:** an auto-generated, human-readable digest per dataset: best RNN, best LSTM, winners by val/test, notable trends (e.g., which factors helped most in Stages 2–3 and by how much), and any non-converged or diverged runs. State numbers only; do not fabricate interpretations.

---

## 8. Coding standards

- Clear, commented, modular code; type hints on public functions; docstrings that state shapes (e.g., `x: (B, T_max, D)`).
- CLI via `argparse` with sensible defaults and `--help` text.
- No dead code, no unexplained magic numbers (constants named at top of file).
- Dependencies in `requirements.txt`: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` (for metrics only), `tqdm` (optional), `seaborn` (optional; or use matplotlib only).
- `setup_guide.md` must list: environment setup, expected data location under `src/data/`, and the **exact command sequence** to reproduce everything:
  1. `python data_loader.py --dataset handwritten` (and `cv`) -> discovery summary + selftest
  2. `python run_experiments.py --dataset handwritten`
  3. `python run_experiments.py --dataset cv`
  4. `python final_eval.py --dataset handwritten` (and `cv`)
  5. `python generate_report_plots.py`
- `contributions.md`: a blank template with a table for team members and tasks.

---

## 9. Pitfalls to explicitly avoid

- Using the `dev`/test data for hyperparameter selection or stopping.
- Fixed-epoch training presented as convergence.
- Letting padding leak into the hidden state.
- Fitting normalization statistics on val/test data.
- Comparing configurations on different train/val splits or a single seed.
- Plotting training curves from the noisy running batch loss when the clean evaluation-pass error is what the convergence criterion uses (the plot must show the same quantity the criterion tested).
- Silent truncation/skipping of malformed files (must be counted and logged).
- Rendering non-Latin class names in plots.

---

## 10. Acceptance checklist (verify at the end of your answer)

- [ ] Four model families run end-to-end: `{handwritten, cv} x {rnn, lstm}`.
- [ ] Convergence uses `|E_t - E_{t-1}| < 1e-4`; stop reasons logged; safety cap clearly separated.
- [ ] Train/val split fixed and shared across all configs; test used only for reporting.
- [ ] Stages 1–4 implemented, multi-seed, mean ± std reported.
- [ ] Items (a)–(d) of the assignment each have a concrete output file as described in Section 7.
- [ ] Folder layout matches Section 2; runs from inside `src/`; resumable.
- [ ] `setup_guide.md`, `contributions.md`, `requirements.txt` included.

**Output format for your answer:** provide each file in full, in its own fenced code block labeled with its relative path, in this order: `requirements.txt`, `utils/*.py`, `data_loader.py`, `models.py`, `train.py`, `run_experiments.py`, `final_eval.py`, `generate_report_plots.py`, `setup_guide.md`, `contributions.md`. Then give a short list of any assumptions you made about the data layout.
