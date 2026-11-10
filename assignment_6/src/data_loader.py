"""Dataset discovery, parsing, normalization, stratified train/val splitting, and DataLoader batching."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from models import SeqClassifier
from utils.io import load_json, save_json, setup_logger
from utils.seed import set_seed

SRC_DIR = Path(__file__).resolve().parent
DATA_ROOT = SRC_DIR / "data"
RESULTS_ROOT = SRC_DIR / "results"
DEFAULT_SPLIT_SEED = 1234
DEFAULT_VAL_RATIO = 0.20
EXPECTED_NUM_CLASSES = 5
CV_FEATURE_DIM = 39
EPS = 1e-8
NORMALIZED_IMAGES_DIRNAME = "normalized_images"
INVERT_Y = False

TRAIN_SPLIT_NAMES = {"train"}
TEST_SPLIT_NAMES = {"dev", "test"}

logger = setup_logger("data_loader")


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for a single sequential data sample."""

    path: Path
    rel_path: str
    class_name: str
    label: int
    split: Literal["train", "test"]


def _resolve_dataset_dir(data_root: Path, dataset: Literal["handwritten", "cv"]) -> Path:
    """Resolve the dataset subdirectory under `data_root`."""
    candidates = (
        [data_root / "1_handwritten", data_root / "handwritten", data_root]
        if dataset == "handwritten"
        else [data_root / "2_CV", data_root / "cv", data_root]
    )
    for cand in candidates:
        if cand.is_dir():
            sub_names = {p.name for p in cand.iterdir() if p.is_dir() and not p.name.startswith(".")}
            if len(sub_names - {NORMALIZED_IMAGES_DIRNAME}) >= EXPECTED_NUM_CLASSES or any(
                s.lower() in TRAIN_SPLIT_NAMES | TEST_SPLIT_NAMES for s in sub_names
            ):
                return cand
    raise FileNotFoundError(
        f"Could not locate dataset '{dataset}' under '{data_root}'. Tried: {[str(c) for c in candidates]}"
    )


def _log_mapping_file(data_root: Path, dataset_dir: Path, dataset: str) -> None:
    """Best-effort log of Mapping_Handwritten.txt / Mapping_CV.txt if present."""
    names = ["Mapping_Handwritten.txt"] if dataset == "handwritten" else ["Mapping_CV.txt"]
    for base in (dataset_dir, data_root, SRC_DIR):
        for name in names:
            mp = base / name
            if mp.is_file():
                text = mp.read_text(encoding="utf-8", errors="replace").strip()
                logger.info("Found %s:\n%s", mp, text)
                return


def _is_valid_data_file(path: Path, dataset: Literal["handwritten", "cv"]) -> bool:
    """Filter out hidden files, mapping files, and rendered images."""
    if not path.is_file() or path.name.startswith("."):
        return False
    if path.name.lower().startswith("mapping_"):
        return False
    suffix = path.suffix.lower()
    if dataset == "handwritten":
        return suffix == ".txt"
    return suffix == ".mfcc" or suffix == ".txt" or suffix == ""


def discover_dataset(
    data_root: str | Path,
    dataset: Literal["handwritten", "cv"],
) -> Tuple[Path, List[str], List[SampleRecord], List[SampleRecord]]:
    """Discover the 5 classes and train/test files under both supported folder nestings.

    Supports:
      - Layout A: `<dataset_dir>/<class>/(train|dev|test)/*`
      - Layout B: `<dataset_dir>/(train|dev|test)/<class>/*`
    """
    root = Path(data_root)
    dataset_dir = _resolve_dataset_dir(root, dataset)
    _log_mapping_file(root, dataset_dir, dataset)

    top_dirs = sorted(
        [
            d
            for d in dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != NORMALIZED_IMAGES_DIRNAME
        ],
        key=lambda p: p.name,
    )
    if not top_dirs:
        raise RuntimeError(f"No class or split subdirectories found inside '{dataset_dir}'.")

    top_lower = {d.name.lower() for d in top_dirs}
    raw_records: List[Tuple[Path, str, Literal["train", "test"]]] = []

    if top_lower & TRAIN_SPLIT_NAMES and top_lower & TEST_SPLIT_NAMES:
        # Layout B: <split>/<class>/<files>
        for split_dir in top_dirs:
            sname = split_dir.name.lower()
            if sname in TRAIN_SPLIT_NAMES:
                canonical_split: Literal["train", "test"] = "train"
            elif sname in TEST_SPLIT_NAMES:
                canonical_split = "test"
            else:
                continue
            for class_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
                if not class_dir.is_dir() or class_dir.name.startswith("."):
                    continue
                for fpath in sorted(class_dir.rglob("*")):
                    if _is_valid_data_file(fpath, dataset):
                        raw_records.append((fpath, class_dir.name, canonical_split))
    else:
        # Layout A: <class>/<split>/<files>
        for class_dir in top_dirs:
            for split_dir in sorted(class_dir.iterdir(), key=lambda p: p.name):
                if not split_dir.is_dir() or split_dir.name.startswith("."):
                    continue
                sname = split_dir.name.lower()
                if sname in TRAIN_SPLIT_NAMES:
                    canonical_split = "train"
                elif sname in TEST_SPLIT_NAMES:
                    canonical_split = "test"
                else:
                    continue
                for fpath in sorted(split_dir.rglob("*")):
                    if _is_valid_data_file(fpath, dataset):
                        raw_records.append((fpath, class_dir.name, canonical_split))

    class_names = sorted({cls_name for _, cls_name, _ in raw_records})
    if len(class_names) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_NUM_CLASSES} classes for '{dataset}' under '{dataset_dir}', "
            f"found {len(class_names)}: {class_names}"
        )

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_records: List[SampleRecord] = []
    test_records: List[SampleRecord] = []

    for fpath, cls_name, split in raw_records:
        rec = SampleRecord(
            path=fpath,
            rel_path=fpath.relative_to(dataset_dir).as_posix(),
            class_name=cls_name,
            label=class_to_idx[cls_name],
            split=split,
        )
        if split == "train":
            train_records.append(rec)
        else:
            test_records.append(rec)

    train_records.sort(key=lambda r: r.rel_path)
    test_records.sort(key=lambda r: r.rel_path)

    # Validate non-empty train and test per class and print discovery summary.
    logger.info("Discovery summary for dataset='%s' at '%s':", dataset, dataset_dir)
    for idx, cls_name in enumerate(class_names):
        n_tr = sum(1 for r in train_records if r.label == idx)
        n_te = sum(1 for r in test_records if r.label == idx)
        logger.info("  [%d] %-6s | train=%4d | test=%4d", idx, cls_name, n_tr, n_te)
        if n_tr == 0 or n_te == 0:
            raise RuntimeError(
                f"Class '{cls_name}' has empty split (train={n_tr}, test={n_te})."
            )
    logger.info("  Total     | train=%4d | test=%4d", len(train_records), len(test_records))

    return dataset_dir, class_names, train_records, test_records


def parse_handwritten_file(path: Path) -> Tuple[np.ndarray, bool]:
    """Parse a handwritten stroke file: first token N followed by N (x, y) pairs.

    Returns:
        (points of shape (T, 2) float32, malformed_flag).
    """
    raw_text = path.read_text(encoding="utf-8", errors="replace").strip()
    tokens = [t for t in re.split(r"[\s,]+", raw_text) if t]
    if len(tokens) < 3:
        raise ValueError(f"Handwritten sample '{path}' has fewer than 3 tokens.")

    n_declared = int(round(float(tokens[0])))
    coord_vals = [float(t) for t in tokens[1:]]
    malformed = False

    if len(coord_vals) != 2 * n_declared:
        malformed = True
        usable = 2 * min(n_declared, len(coord_vals) // 2) if n_declared > 0 else 2 * (len(coord_vals) // 2)
        if usable <= 0:
            usable = 2 * (len(coord_vals) // 2)
        coord_vals = coord_vals[:usable]
        logger.warning(
            "Malformed handwritten sample '%s': declared N=%d (expected %d values), found %d values; trimmed to %d.",
            path,
            n_declared,
            2 * n_declared,
            len(tokens) - 1,
            usable,
        )

    if len(coord_vals) < 2:
        raise ValueError(f"Handwritten sample '{path}' contains zero valid (x, y) pairs.")

    pts = np.asarray(coord_vals, dtype=np.float32).reshape(-1, 2)
    return pts, malformed


def normalize_handwritten_xy(seq: np.ndarray) -> np.ndarray:
    """Min-max normalize x and y coordinates independently per sample into [0, 1]."""
    out = seq.astype(np.float32, copy=True)
    for col in range(2):
        cmin = float(out[:, col].min())
        cmax = float(out[:, col].max())
        span = cmax - cmin
        if span < EPS:
            out[:, col] = 0.0
        else:
            out[:, col] = np.clip((out[:, col] - cmin) / span, 0.0, 1.0)
    return out


def parse_cv_file(path: Path) -> Tuple[np.ndarray, bool]:
    """Parse a CV MFCC file of shape (T, 39) tolerating whitespace/comma delimiters."""
    rows: List[List[float]] = []
    malformed = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        vals = [float(tok) for tok in re.split(r"[\s,]+", stripped) if tok]
        if len(vals) != CV_FEATURE_DIM:
            malformed = True
            if len(vals) > CV_FEATURE_DIM:
                vals = vals[:CV_FEATURE_DIM]
            else:
                vals = vals + [0.0] * (CV_FEATURE_DIM - len(vals))
        rows.append(vals)

    if malformed:
        logger.warning("Malformed CV sample '%s': encountered row(s) with != %d columns.", path, CV_FEATURE_DIM)
    if not rows:
        raise ValueError(f"CV sample '{path}' is empty.")

    return np.asarray(rows, dtype=np.float32), malformed


def compute_train_standardization(train_seqs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise mean and std strictly across training sequences."""
    stacked = np.concatenate(train_seqs, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardization(seqs: Sequence[np.ndarray], mean: np.ndarray, std: np.ndarray) -> List[np.ndarray]:
    """Apply precomputed training-split standardization to a list of sequences."""
    return [((s - mean) / std).astype(np.float32) for s in seqs]


def load_or_create_split(
    train_records: Sequence[SampleRecord],
    split_path: Path,
    val_ratio: float = DEFAULT_VAL_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[int], List[int]]:
    """Create or reload a fixed stratified train/val split over training records only."""
    if split_path.is_file():
        saved = load_json(split_path)
        expected_rels = [r.rel_path for r in train_records]
        if (
            saved.get("split_seed") == split_seed
            and abs(float(saved.get("val_ratio", -1.0)) - val_ratio) < 1e-9
            and saved.get("all_train_rel_paths") == expected_rels
        ):
            return list(saved["train_indices"]), list(saved["val_indices"])
        logger.warning("Existing split file '%s' differs in seed/ratio/files; regenerating.", split_path)

    labels = np.array([r.label for r in train_records], dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=split_seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    train_indices = sorted(int(i) for i in train_idx)
    val_indices = sorted(int(i) for i in val_idx)

    save_json(
        {
            "split_seed": split_seed,
            "val_ratio": val_ratio,
            "num_train_total": len(train_records),
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_rel_paths": [train_records[i].rel_path for i in train_indices],
            "val_rel_paths": [train_records[i].rel_path for i in val_indices],
            "all_train_rel_paths": [r.rel_path for r in train_records],
        },
        split_path,
    )
    logger.info("Saved stratified split (%d train / %d val) to '%s'.", len(train_indices), len(val_indices), split_path)
    return train_indices, val_indices


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping variable-length numpy sequences and integer labels."""

    def __init__(self, sequences: Sequence[np.ndarray], labels: Sequence[int]) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have equal length.")
        self.sequences = [torch.as_tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx]


def pad_collate_fn(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad variable-length sequences in a batch to (B, T_max, D), returning (x, lengths, labels)."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.int64)
    feat_dim = seqs[0].size(1)
    max_len = int(lengths.max().item())

    padded = torch.zeros((len(seqs), max_len, feat_dim), dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0), :] = s

    label_tensor = torch.tensor(labels, dtype=torch.int64)
    return padded, lengths, label_tensor


@dataclass
class PreparedData:
    """Container holding prepared train/val/test DataLoaders and dataset metadata."""

    dataset_name: Literal["handwritten", "cv"]
    dataset_dir: Path
    class_names: List[str]
    input_size: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_eval_loader: DataLoader
    scaler_mean: Optional[List[float]]
    scaler_std: Optional[List[float]]
    malformed_count: int


def _load_records(
    records: Sequence[SampleRecord],
    dataset: Literal["handwritten", "cv"],
) -> Tuple[List[np.ndarray], List[int], int]:
    """Parse and normalize a list of sample records."""
    seqs: List[np.ndarray] = []
    labels: List[int] = []
    malformed_total = 0

    for rec in records:
        if dataset == "handwritten":
            raw, bad = parse_handwritten_file(rec.path)
            proc = normalize_handwritten_xy(raw)
        else:
            proc, bad = parse_cv_file(rec.path)
        if bad:
            malformed_total += 1
        seqs.append(proc)
        labels.append(rec.label)

    return seqs, labels, malformed_total


def prepare_dataloaders(
    dataset: Literal["handwritten", "cv"],
    data_root: str | Path = DATA_ROOT,
    results_root: str | Path = RESULTS_ROOT,
    batch_size: int = 32,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = 42,
) -> PreparedData:
    """Discover, parse, split, normalize/standardize, and wrap datasets into DataLoaders."""
    dataset_dir, class_names, all_train_records, test_records = discover_dataset(data_root, dataset)
    split_path = Path(results_root) / dataset / "split.json"
    train_idx, val_idx = load_or_create_split(
        all_train_records, split_path=split_path, val_ratio=val_ratio, split_seed=split_seed
    )

    tr_records = [all_train_records[i] for i in train_idx]
    va_records = [all_train_records[i] for i in val_idx]

    tr_seqs, tr_labels, bad_tr = _load_records(tr_records, dataset)
    va_seqs, va_labels, bad_va = _load_records(va_records, dataset)
    te_seqs, te_labels, bad_te = _load_records(test_records, dataset)

    malformed_total = bad_tr + bad_va + bad_te
    if malformed_total > 0:
        logger.warning("Total malformed files across '%s': %d", dataset, malformed_total)
    else:
        logger.info("Parsed all files cleanly for '%s' (0 malformed files).", dataset)

    scaler_mean_list: Optional[List[float]] = None
    scaler_std_list: Optional[List[float]] = None
    if dataset == "cv":
        mean_arr, std_arr = compute_train_standardization(tr_seqs)
        tr_seqs = apply_standardization(tr_seqs, mean_arr, std_arr)
        va_seqs = apply_standardization(va_seqs, mean_arr, std_arr)
        te_seqs = apply_standardization(te_seqs, mean_arr, std_arr)
        scaler_mean_list = mean_arr.tolist()
        scaler_std_list = std_arr.tolist()

    input_size = int(tr_seqs[0].shape[1])

    tr_ds = SequenceDataset(tr_seqs, tr_labels)
    va_ds = SequenceDataset(va_seqs, va_labels)
    te_ds = SequenceDataset(te_seqs, te_labels)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        generator=generator,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        tr_ds, batch_size=max(batch_size, 512), shuffle=False, collate_fn=pad_collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        va_ds, batch_size=max(batch_size, 512), shuffle=False, collate_fn=pad_collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        te_ds, batch_size=max(batch_size, 512), shuffle=False, collate_fn=pad_collate_fn, num_workers=0
    )

    return PreparedData(
        dataset_name=dataset,
        dataset_dir=dataset_dir,
        class_names=class_names,
        input_size=input_size,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_eval_loader=train_eval_loader,
        scaler_mean=scaler_mean_list,
        scaler_std=scaler_std_list,
        malformed_count=malformed_total,
    )


def render_stroke_on_axis(
    ax: plt.Axes,
    norm_pts: np.ndarray,
    invert_y: bool = INVERT_Y,
    title: Optional[str] = None,
) -> None:
    """Draw normalized [0, 1] handwritten stroke polyline with hairline edges and red vertex markers."""
    x = norm_pts[:, 0]
    y = norm_pts[:, 1]
    ax.plot(x, y, color="#1f77b4", linewidth=0.5, zorder=2)
    ax.scatter(x, y, color="#d62728", s=5.0, linewidths=0.0, zorder=3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    if title:
        ax.set_title(title, fontsize=9)


def export_all_handwritten_normalized_images(
    data_root: str | Path = DATA_ROOT,
    invert_y: bool = INVERT_Y,
    overwrite: bool = True,
) -> Path:
    """Render and save the normalized image of EVERY handwritten sample inside `normalized_images/`.

    Images are saved under:
      `<dataset_dir>/normalized_images/<class_name>/<split>/<sample_stem>.png`
    so the raw `.txt` directories remain untouched while all normalized representations are stored
    separately within the handwritten dataset folder.
    """
    dataset_dir, _, train_records, test_records = discover_dataset(data_root, "handwritten")
    out_root = dataset_dir / NORMALIZED_IMAGES_DIRNAME
    all_records = list(train_records) + list(test_records)

    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=150)
    saved_count = 0
    for rec in all_records:
        out_path = out_root / rec.class_name / rec.split / f"{rec.path.stem}.png"
        if out_path.is_file() and not overwrite:
            saved_count += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw, _ = parse_handwritten_file(rec.path)
        norm_pts = normalize_handwritten_xy(raw)
        ax.clear()
        render_stroke_on_axis(
            ax,
            norm_pts,
            invert_y=invert_y,
            title=f"{rec.class_name} ({rec.split}/{rec.path.stem})",
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        saved_count += 1

    plt.close(fig)
    logger.info(
        "Exported %d normalized handwritten images to '%s' (invert_y=%s).",
        saved_count,
        out_root,
        invert_y,
    )
    return out_root


def run_selftest(dataset: Literal["handwritten", "cv"], data_root: str | Path = DATA_ROOT) -> None:
    """Verify padding invariance and normalization bounds."""
    set_seed(42)
    prep = prepare_dataloaders(dataset=dataset, data_root=data_root, batch_size=8, seed=42)

    if dataset == "handwritten":
        for loader in (prep.train_loader, prep.val_loader, prep.test_loader):
            for seq in loader.dataset.sequences:
                assert seq.shape[1] == 2, f"Expected 2D sequence, got {seq.shape}"
                assert float(seq.min()) >= -1e-6 and float(seq.max()) <= 1.0 + 1e-6, (
                    f"Handwritten coordinates out of [0, 1]: [{seq.min()}, {seq.max()}]"
                )

    sample_x, sample_lengths, _ = next(iter(prep.train_loader))
    device = torch.device("cpu")
    for rnn_type in ("rnn", "lstm"):
        for bidir in (False, True):
            model = SeqClassifier(
                rnn_type=rnn_type,
                input_size=prep.input_size,
                hidden_size=16,
                num_layers=2,
                bidirectional=bidir,
                num_classes=len(prep.class_names),
            ).to(device)
            model.eval()
            with torch.no_grad():
                batch_logits = model(sample_x.to(device), sample_lengths.to(device))
                for b in range(min(3, sample_x.size(0))):
                    t_len = int(sample_lengths[b].item())
                    single_x = sample_x[b : b + 1, :t_len, :].to(device)
                    single_len = torch.tensor([t_len], dtype=torch.int64, device=device)
                    single_logits = model(single_x, single_len)
                    max_diff = float((batch_logits[b : b + 1] - single_logits).abs().max().item())
                    assert max_diff < 1e-5, f"Padding leak detected ({rnn_type}, bidir={bidir}): max_diff={max_diff}"
    logger.info("Selftest PASSED for dataset='%s': zero padding leakage.", dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset discovery, preprocessing, and selftest for Assignment 6.")
    parser.add_argument("--dataset", type=str, required=True, choices=["handwritten", "cv"])
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--selftest", action="store_true", help="Run padding invariance and normalization checks.")
    parser.add_argument(
        "--export_all_images",
        action="store_true",
        help="Export normalized PNG image for every handwritten sample into normalized_images/.",
    )
    parser.add_argument("--invert_y", action="store_true", default=INVERT_Y)
    args = parser.parse_args()

    run_selftest(dataset=args.dataset, data_root=args.data_root)
    if args.dataset == "handwritten" and args.export_all_images:
        export_all_handwritten_normalized_images(data_root=args.data_root, invert_y=args.invert_y)


if __name__ == "__main__":
    main()
