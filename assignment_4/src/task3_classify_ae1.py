# Task 3: Classification using compressed representations extracted from 1-hidden-layer autoencoders, FCNN training, and comparison with Assignment 3 and Task 1.

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
from models import ARCH_GROUPS, build_autoencoder
from train import run_classification_study, default_device
from utils.plotting import plot_accuracy_vs_dimension, plot_val_accuracy_heatmap
from utils.parallel_study import save_codes, make_jobs, pretrain_in_parallel, default_workers
from utils import training_config as cfg

BOTTLENECKS = [32, 64, 128, 256]
DEFAULT_AE_DIR = os.path.join(CURRENT_DIR, "results", "task2_autoencoders")
DEFAULT_RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "task3_classify_ae1")
TASK1_DIR = os.path.join(CURRENT_DIR, "results", "task1_pca")


def parse_args():
    p = argparse.ArgumentParser(description="Task 3: FCNN classification on 1-hidden-layer AE codes")
    p.add_argument("--dims", type=int, nargs="+", default=BOTTLENECKS, help="bottleneck sizes (default: 32 64 128 256)")
    p.add_argument("--archs", type=str, nargs="+", default=None,
                   help="Architectures: keys from models.ARCHITECTURES or groups (3_layers/4_layers/5_layers/all). "
                        "Default: all 9 (the same ones as Task 1)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--ae_dir", type=str, default=DEFAULT_AE_DIR, help="Task 2 output folder holding ae1_bottleneck_<k>/model.pt")
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="1 = SGD as in Task 1; 0 = full batch")
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--momentum", type=float, default=cfg.MOMENTUM)
    p.add_argument("--max_epochs", type=int, default=cfg.MAX_EPOCHS)
    p.add_argument("--threshold", type=float, default=cfg.STOPPING_THRESHOLD)
    p.add_argument("--patience", type=int, default=cfg.PATIENCE)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--device", type=str, default=None, help="cpu / cuda (default: cpu for batch size 1)")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel CPU worker processes for the independent FCNN trainings "
                        "(0 = auto = number of CPU cores, 1 = sequential). Only used on cpu.")
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
    """Loads the Task 2 ae1 autoencoder for bottleneck k (on CPU) in eval mode."""
    path = os.path.join(ae_dir, f"ae1_bottleneck_{k}", "model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Task 2 checkpoint: {path}\nRun task2_autoencoders.py first.")
    model = build_autoencoder("ae1", k)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def encode(model, X, chunk=4096):
    """Output of the (linear) middle layer for every row of X."""
    return torch.cat([model.encode(X[i:i + chunk]) for i in range(0, len(X), chunk)])


# ------------------------------------------------------------------ comparison helpers

def load_task1_results():
    """{dim: result dict} from Task 1 (if it has been run)."""
    try:
        with open(os.path.join(TASK1_DIR, "summary", "task1_results.json")) as f:
            return {r["input_dim"]: r for r in json.load(f)["results"]}
    except (OSError, ValueError, KeyError):
        return {}


# ------------------------------------------------------------------ summary

def write_summary(results, archs, results_dir):
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    dims = [r["input_dim"] for r in results]
    ref = cfg.A3_REFERENCE
    t1 = load_task1_results()

    # ---- (c) best bottleneck size, by test accuracy ----
    best = max(results, key=lambda r: r["test_acc"])

    lines = ["Task 3 summary: 1-hidden-layer autoencoder codes + FCNN classification", "=" * 78,
             f"{'Dim':>5} {'Best arch (by val acc)':<24} {'Val acc':>8} {'Test acc':>9} {'Macro F1':>9}",
             "-" * 78]
    for r in results:
        lines.append(f"{r['input_dim']:>5} {r['best_arch']:<24} "
                     f"{r['best_val_acc']:>7.2f}% {r['test_acc']:>8.2f}% {r['test_macro_f1']:>8.2f}%")
    lines += ["", f"(c) Best reduced dimension (highest test accuracy): {best['input_dim']}  "
                  f"-> {best['test_acc']:.2f}% test accuracy with {best['best_arch']} {best['best_hidden_dims']}",
              "", "(d) Comparison (test accuracy, %):",
              f"    Assignment 3 (raw 784 pixels, {ref['architecture']}): {ref['test_acc']:.2f}",
              f"    {'Dim':>5} {'Task 3 (AE-1)':>14} {'Task 1 (PCA)':>14} {'vs A3':>8} {'vs Task 1':>10}"]
    for r in results:
        k = r["input_dim"]
        p1 = f"{t1[k]['test_acc']:.2f}" if k in t1 else "n/a"
        d1 = f"{r['test_acc'] - t1[k]['test_acc']:+.2f}" if k in t1 else "n/a"
        lines.append(f"    {k:>5} {r['test_acc']:>14.2f} {p1:>14} {r['test_acc'] - ref['test_acc']:>+8.2f} {d1:>10}")
    if not t1:
        lines.append("    (Task 1 results not found; run task1_pca.py and re-run this script to fill that column. "
                     "Finished classifiers are resumed from checkpoints, so the re-run is quick.)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(summary_dir, "task3_summary.txt"), "w") as f:
        f.write(text + "\n")

    with open(os.path.join(summary_dir, "task3_summary.csv"), "w") as f:
        f.write("dimension,best_architecture,hidden_layers,val_acc_pct,test_acc_pct,test_macro_f1_pct,"
                "delta_test_acc_vs_assignment3,task1_pca_test_acc_pct,delta_test_acc_vs_task1\n")
        for r in results:
            k = r["input_dim"]
            p1 = f"{t1[k]['test_acc']:.2f}" if k in t1 else ""
            d1 = f"{r['test_acc'] - t1[k]['test_acc']:+.2f}" if k in t1 else ""
            f.write(f"{k},{r['best_arch']},\"{'-'.join(map(str, r['best_hidden_dims']))}\","
                    f"{r['best_val_acc']:.2f},{r['test_acc']:.2f},{r['test_macro_f1']:.2f},"
                    f"{r['test_acc'] - ref['test_acc']:+.2f},{p1},{d1}\n")

    # validation accuracy of every architecture at every bottleneck size (Step b.ii at a glance)
    mat = [[r["val_acc_by_arch"].get(a, float("nan")) for r in results] for a in archs]
    with open(os.path.join(summary_dir, "validation_accuracy_all.csv"), "w") as f:
        f.write("architecture," + ",".join(f"AE1-{d}" for d in dims) + "\n")
        for a, row in zip(archs, mat):
            f.write(a + "," + ",".join("" if np.isnan(v) else f"{v:.2f}" for v in row) + "\n")

    plot_val_accuracy_heatmap(archs, dims, mat,
                              title="Task 3: validation accuracy (%) per architecture and AE-1 bottleneck",
                              filename=os.path.join(summary_dir, "val_acc_heatmap.png"))
    plot_accuracy_vs_dimension(dims, [r["test_acc"] for r in results], reference=ref["test_acc"],
                               reference_label="Assignment 3 (784-dim)",
                               title="Task 3: test accuracy vs AE-1 bottleneck size",
                               filename=os.path.join(summary_dir, "test_acc_vs_dim.png"))

    with open(os.path.join(summary_dir, "task3_results.json"), "w") as f:
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
    num_classes = len(idx_to_class)

    # ---- (a) compressed representations of train / val / test (middle layer output) ----
    codes = {}
    for k in args.dims:
        ae = load_ae(args.ae_dir, k)
        Ztr, Zva, Zte = encode(ae, Xtr).float(), encode(ae, Xva).float(), encode(ae, Xte).float()
        out_dir = os.path.join(args.results_dir, f"ae1_{k}")
        codes_path = os.path.join(out_dir, "codes.npz")
        save_codes(codes_path, Ztr, ytr, Zva, yva, Zte, yte)          # "save the output of the middle layer"
        codes[k] = (Ztr, Zva, Zte, out_dir, codes_path)
        print(f"AE-1 bottleneck {k}: codes train {tuple(Ztr.shape)}, val {tuple(Zva.shape)}, "
              f"test {tuple(Zte.shape)}  (saved to {codes_path})")

    # ---- (b) FCNN trainings. They are independent, so on CPU run them all in a worker pool first;
    #          run_classification_study below then just picks up the finished checkpoints. ----
    hyper = dict(batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, max_epochs=args.max_epochs,
                 threshold=args.threshold, patience=args.patience, seed=args.seed, activation=cfg.ACTIVATION)
    workers = args.workers or default_workers(len(args.dims) * len(archs))
    study_resume = not args.no_resume
    if device == "cpu" and workers > 1:
        jobs = []
        for k in args.dims:
            jobs += make_jobs(codes[k][4], codes[k][3], archs, num_classes, hyper)
        pretrain_in_parallel(jobs, workers, resume=not args.no_resume)
        study_resume = True                       # load what the workers just trained

    results = []
    for k in args.dims:
        Ztr, Zva, Zte, out_dir, _ = codes[k]
        res = run_classification_study(
            name=f"AE1-{k}", X_train=Ztr.to(device), y_train=ytr.to(device), X_val=Zva.to(device),
            y_val=yva.to(device), X_test=Zte.to(device), y_test=yte.to(device), idx_to_class=idx_to_class,
            out_dir=out_dir, archs=archs,
            batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, max_epochs=args.max_epochs,
            threshold=args.threshold, patience=args.patience, seed=args.seed, resume=study_resume)
        results.append(res)

    write_summary(results, archs, args.results_dir)


if __name__ == "__main__":
    main()
