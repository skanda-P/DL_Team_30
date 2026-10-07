# Task 1: PCA dimension reduction (32, 64, 128, 256), FCNN classification on reduced representations, and comparison with Assignment 3.

import os
import sys
import json
import argparse
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import get_data_tensors
from models import ARCH_GROUPS
from train import run_classification_study, default_device
from utils.pca import PCA
from utils.plotting import plot_accuracy_vs_dimension, plot_val_accuracy_heatmap
from utils import training_config as cfg

DEFAULT_RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "task1_pca")


def parse_args():
    p = argparse.ArgumentParser(description="Task 1: PCA + FCNN classification")
    p.add_argument("--dims", type=int, nargs="+", default=cfg.PCA_DIMS, help="PCA dimensions (default: 32 64 128 256)")
    p.add_argument("--archs", type=str, nargs="+", default=None,
                   help="Architectures: keys from models.ARCHITECTURES or groups (3_layers/4_layers/5_layers/all). Default: all")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="1 = SGD as in Assignment 3; 0 = full batch")
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


def main():
    args = parse_args()
    device = args.device or default_device(args.batch_size)
    archs = resolve_archs(args.archs)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device} | dims: {args.dims} | architectures: {len(archs)}")
    (Xtr, ytr), (Xva, yva), (Xte, yte), class_to_idx, idx_to_class = get_data_tensors(args.data_dir, device="cpu")
    print(f"Loaded: train {tuple(Xtr.shape)}, val {tuple(Xva.shape)}, test {tuple(Xte.shape)}, classes {class_to_idx}")

    # ---- (a) PCA: eigenvectors and mean from the TRAINING set only ----
    pca = PCA().fit(Xtr.numpy())

    results = []
    for k in args.dims:
        Ztr = torch.from_numpy(pca.transform(Xtr.numpy(), k)).float().to(device)
        Zva = torch.from_numpy(pca.transform(Xva.numpy(), k)).float().to(device)   # train mean + train eigenvectors
        Zte = torch.from_numpy(pca.transform(Xte.numpy(), k)).float().to(device)

        res = run_classification_study(
            name=f"PCA-{k}", X_train=Ztr, y_train=ytr.to(device), X_val=Zva, y_val=yva.to(device),
            X_test=Zte, y_test=yte.to(device), idx_to_class=idx_to_class,
            out_dir=os.path.join(args.results_dir, f"pca_{k}"), archs=archs,
            batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, max_epochs=args.max_epochs,
            threshold=args.threshold, patience=args.patience, seed=args.seed, resume=not args.no_resume)
        res["explained_variance_pct"] = pca.explained_variance_ratio(k) * 100
        results.append(res)

    write_summary(results, archs, args.results_dir)


def write_summary(results, archs, results_dir):
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    dims = [r["input_dim"] for r in results]
    ref = cfg.A3_REFERENCE

    # ---- (c) best reduced dimension, by test accuracy ----
    best = max(results, key=lambda r: r["test_acc"])

    lines = ["Task 1 summary: PCA + FCNN classification", "=" * 78,
             f"{'Dim':>5} {'Var kept':>9} {'Best arch (by val acc)':<24} {'Val acc':>8} {'Test acc':>9} {'Macro F1':>9}",
             "-" * 78]
    for r in results:
        lines.append(f"{r['input_dim']:>5} {r['explained_variance_pct']:>8.2f}% {r['best_arch']:<24} "
                     f"{r['best_val_acc']:>7.2f}% {r['test_acc']:>8.2f}% {r['test_macro_f1']:>8.2f}%")
    lines += ["", f"(c) Best reduced dimension (highest test accuracy): {best['input_dim']}  "
                  f"-> {best['test_acc']:.2f}% test accuracy with {best['best_arch']} {best['best_hidden_dims']}",
              "", "(d) Comparison with the best result of Assignment 3 (raw 784-dim pixels):",
              f"    Assignment 3: {ref['architecture']} {ref['hidden_dims']}, {ref['optimizer']}",
              f"    Assignment 3: val {ref['val_acc']:.2f}%  test {ref['test_acc']:.2f}%  macro F1 {ref['test_macro_f1']:.2f}%"]
    for r in results:
        delta = r["test_acc"] - ref["test_acc"]
        lines.append(f"    PCA-{r['input_dim']:<4}: test {r['test_acc']:.2f}%  ({delta:+.2f} pts vs Assignment 3)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(summary_dir, "task1_summary.txt"), "w") as f:
        f.write(text + "\n")

    with open(os.path.join(summary_dir, "task1_summary.csv"), "w") as f:
        f.write("dimension,variance_retained_pct,best_architecture,hidden_layers,val_acc_pct,test_acc_pct,test_macro_f1_pct,"
                "delta_test_acc_vs_assignment3\n")
        for r in results:
            f.write(f"{r['input_dim']},{r['explained_variance_pct']:.2f},{r['best_arch']},"
                    f"\"{'-'.join(map(str, r['best_hidden_dims']))}\",{r['best_val_acc']:.2f},{r['test_acc']:.2f},"
                    f"{r['test_macro_f1']:.2f},{r['test_acc'] - ref['test_acc']:+.2f}\n")

    # validation accuracy of every architecture at every dimension (Step b.ii at a glance)
    mat = [[r["val_acc_by_arch"].get(a, float("nan")) for r in results] for a in archs]
    with open(os.path.join(summary_dir, "validation_accuracy_all.csv"), "w") as f:
        f.write("architecture," + ",".join(f"PCA-{d}" for d in dims) + "\n")
        for a, row in zip(archs, mat):
            f.write(a + "," + ",".join("" if np.isnan(v) else f"{v:.2f}" for v in row) + "\n")

    plot_val_accuracy_heatmap(archs, dims, mat, title="Task 1: validation accuracy (%) per architecture and PCA dimension",
                              filename=os.path.join(summary_dir, "val_acc_heatmap.png"))
    plot_accuracy_vs_dimension(dims, [r["test_acc"] for r in results], reference=ref["test_acc"],
                               reference_label="Assignment 3 (784-dim)",
                               title="Task 1: test accuracy vs PCA dimension",
                               filename=os.path.join(summary_dir, "test_acc_vs_dim.png"))

    with open(os.path.join(summary_dir, "task1_results.json"), "w") as f:
        json.dump({"results": results, "assignment3_reference": ref, "best_dimension": best["input_dim"]}, f, indent=2)
    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()