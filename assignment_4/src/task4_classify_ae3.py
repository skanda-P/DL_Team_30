# Task 4: Classification using compressed representations extracted from deep (3-hidden-layer) autoencoders, FCNN training, and comparison with Assignment 3, Task 1, and Task 3.

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import get_data_tensors
from models import ARCH_GROUPS, build_autoencoder
from train import run_classification_study, default_device
from utils.plotting import plot_accuracy_vs_dimension, plot_val_accuracy_heatmap
from utils import training_config as cfg

BOTTLENECKS = [32, 64, 128, 256]
DEFAULT_AE_DIR = os.path.join(CURRENT_DIR, "results", "task2_autoencoders")
DEFAULT_RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "task4_classify_ae3")
TASK1_DIR = os.path.join(CURRENT_DIR, "results", "task1_pca")
TASK3_DIR = os.path.join(CURRENT_DIR, "results", "task3_classify_ae1")


def parse_args():
    p = argparse.ArgumentParser(description="Task 4: FCNN classification on 3-hidden-layer AE codes")
    p.add_argument("--dims", type=int, nargs="+", default=BOTTLENECKS, help="bottleneck sizes (default: 32 64 128 256)")
    p.add_argument("--archs", type=str, nargs="+", default=None,
                   help="Architectures: keys from models.ARCHITECTURES or groups (3_layers/4_layers/5_layers/all). Default: all")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--ae_dir", type=str, default=DEFAULT_AE_DIR, help="Task 2 output folder holding ae3_bottleneck_<k>/model.pt")
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="1 = SGD as in Task 1; 0 = full batch")
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--momentum", type=float, default=cfg.MOMENTUM)
    p.add_argument("--max_epochs", type=int, default=cfg.MAX_EPOCHS)
    p.add_argument("--threshold", type=float, default=cfg.STOPPING_THRESHOLD)
    p.add_argument("--patience", type=int, default=cfg.PATIENCE)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--device", type=str, default=None, help="cpu / cuda (default: cpu for batch size 1)")
    p.add_argument("--no_resume", action="store_true", help="retrain even if checkpoints exist")
    return p.parse_args()


def resolve_archs(names):
    if not names:
        return ARCH_GROUPS["all"]
    out = []
    for n in names:
        out.extend(ARCH_GROUPS[n] if n in ARCH_GROUPS else [n])
    return out


# ------------------------------------------------------------------ (a) encoding

def load_ae(ae_dir, k):
    """Loads the Task 2 ae3 autoencoder for bottleneck k (on CPU) in eval mode."""
    path = os.path.join(ae_dir, f"ae3_bottleneck_{k}", "model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Task 2 checkpoint: {path}\nRun task2_autoencoders.py first.")
    model = build_autoencoder("ae3", k)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def encode(model, X, chunk=4096):
    """Output of the (linear) middle layer for every row of X."""
    return torch.cat([model.encode(X[i:i + chunk]) for i in range(0, len(X), chunk)])


# ------------------------------------------------------------------ comparison helpers

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_task1_results():
    """{dim: result dict} from Task 1 (if it has been run)."""
    data = _load_json(os.path.join(TASK1_DIR, "summary", "task1_results.json"))
    return {r["input_dim"]: r for r in data["results"]} if data else {}


def load_task3_results():
    """{dim: result dict} from Task 3, read from any per-dimension study_result.json it wrote."""
    out = {}
    for path in glob.glob(os.path.join(TASK3_DIR, "**", "study_result.json"), recursive=True):
        r = _load_json(path)
        if r and "test_acc" in r and "input_dim" in r:
            out[r["input_dim"]] = r
    return out


# ------------------------------------------------------------------ summary

def write_summary(results, archs, results_dir):
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    dims = [r["input_dim"] for r in results]
    ref = cfg.A3_REFERENCE
    t1, t3 = load_task1_results(), load_task3_results()

    # ---- (c) best bottleneck size, by test accuracy ----
    best = max(results, key=lambda r: r["test_acc"])

    lines = ["Task 4 summary: 3-hidden-layer autoencoder codes + FCNN classification", "=" * 78,
             f"{'Dim':>5} {'Best arch (by val acc)':<24} {'Val acc':>8} {'Test acc':>9} {'Macro F1':>9}",
             "-" * 78]
    for r in results:
        lines.append(f"{r['input_dim']:>5} {r['best_arch']:<24} "
                     f"{r['best_val_acc']:>7.2f}% {r['test_acc']:>8.2f}% {r['test_macro_f1']:>8.2f}%")
    lines += ["", f"(c) Best reduced dimension (highest test accuracy): {best['input_dim']}  "
                  f"-> {best['test_acc']:.2f}% test accuracy with {best['best_arch']} {best['best_hidden_dims']}",
              "", "(d) Comparison (test accuracy, %):",
              f"    Assignment 3 (raw 784 pixels, {ref['architecture']}): {ref['test_acc']:.2f}",
              f"    {'Dim':>5} {'Task 4 (AE-3)':>14} {'Task 1 (PCA)':>14} {'Task 3 (AE-1)':>14} {'vs A3':>8}"]
    for r in results:
        k = r["input_dim"]
        p1 = f"{t1[k]['test_acc']:.2f}" if k in t1 else "n/a"
        p3 = f"{t3[k]['test_acc']:.2f}" if k in t3 else "n/a"
        lines.append(f"    {k:>5} {r['test_acc']:>14.2f} {p1:>14} {p3:>14} {r['test_acc'] - ref['test_acc']:>+8.2f}")
    if not t3:
        lines.append("    (Task 3 results not found yet. Re-run this script after Task 3 finishes to fill the column;"
                     " finished classifiers are resumed from checkpoints, so it is quick.)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(summary_dir, "task4_summary.txt"), "w") as f:
        f.write(text + "\n")

    with open(os.path.join(summary_dir, "task4_summary.csv"), "w") as f:
        f.write("dimension,best_architecture,hidden_layers,val_acc_pct,test_acc_pct,test_macro_f1_pct,"
                "delta_test_acc_vs_assignment3,task1_pca_test_acc_pct,task3_ae1_test_acc_pct\n")
        for r in results:
            k = r["input_dim"]
            p1 = f"{t1[k]['test_acc']:.2f}" if k in t1 else ""
            p3 = f"{t3[k]['test_acc']:.2f}" if k in t3 else ""
            f.write(f"{k},{r['best_arch']},\"{'-'.join(map(str, r['best_hidden_dims']))}\","
                    f"{r['best_val_acc']:.2f},{r['test_acc']:.2f},{r['test_macro_f1']:.2f},"
                    f"{r['test_acc'] - ref['test_acc']:+.2f},{p1},{p3}\n")

    mat = [[r["val_acc_by_arch"].get(a, float("nan")) for r in results] for a in archs]
    with open(os.path.join(summary_dir, "validation_accuracy_all.csv"), "w") as f:
        f.write("architecture," + ",".join(f"AE3-{d}" for d in dims) + "\n")
        for a, row in zip(archs, mat):
            f.write(a + "," + ",".join("" if np.isnan(v) else f"{v:.2f}" for v in row) + "\n")

    plot_val_accuracy_heatmap(archs, dims, mat,
                              title="Task 4: validation accuracy (%) per architecture and AE-3 bottleneck",
                              filename=os.path.join(summary_dir, "val_acc_heatmap.png"))
    plot_accuracy_vs_dimension(dims, [r["test_acc"] for r in results], reference=ref["test_acc"],
                               reference_label="Assignment 3 (784-dim)",
                               title="Task 4: test accuracy vs AE-3 bottleneck size",
                               filename=os.path.join(summary_dir, "test_acc_vs_dim.png"))

    with open(os.path.join(summary_dir, "task4_results.json"), "w") as f:
        json.dump({"results": results, "assignment3_reference": ref, "best_dimension": best["input_dim"]},
                  f, indent=2)
    print(f"\nSaved results to {results_dir}")


# ------------------------------------------------------------------ main

def main():
    args = parse_args()
    device = args.device or default_device(args.batch_size)
    archs = resolve_archs(args.archs)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device} | bottlenecks: {args.dims} | architectures: {len(archs)}")
    (Xtr, ytr), (Xva, yva), (Xte, yte), class_to_idx, idx_to_class = get_data_tensors(args.data_dir, device="cpu")
    print(f"Loaded: train {tuple(Xtr.shape)}, val {tuple(Xva.shape)}, test {tuple(Xte.shape)}, classes {class_to_idx}")

    results = []
    for k in args.dims:
        ae = load_ae(args.ae_dir, k)
        # (a) compressed representations of train / val / test (middle layer output)
        Ztr = encode(ae, Xtr).float().to(device)
        Zva = encode(ae, Xva).float().to(device)
        Zte = encode(ae, Xte).float().to(device)
        print(f"\nAE-3 bottleneck {k}: codes train {tuple(Ztr.shape)}, val {tuple(Zva.shape)}, test {tuple(Zte.shape)}")

        # (b) same FCNN study as Task 1 (9 architectures, val-based selection, test on best only)
        res = run_classification_study(
            name=f"AE3-{k}", X_train=Ztr, y_train=ytr.to(device), X_val=Zva, y_val=yva.to(device),
            X_test=Zte, y_test=yte.to(device), idx_to_class=idx_to_class,
            out_dir=os.path.join(args.results_dir, f"ae3_{k}"), archs=archs,
            batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, max_epochs=args.max_epochs,
            threshold=args.threshold, patience=args.patience, seed=args.seed, resume=not args.no_resume)
        results.append(res)

    write_summary(results, archs, args.results_dir)


if __name__ == "__main__":
    main()