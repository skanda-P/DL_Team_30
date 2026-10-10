# Task 5: Denoising autoencoders with 20% and 40% noise, reconstruction error analysis, sample denoising visualization, and latent classification.
"""
Task 5 - denoising autoencoders (one hidden layer).

  a) Build 1-hidden-layer denoising autoencoders for 20% and 40% noise. The bottleneck size is the best
     reduced dimension of Task 3 (highest test accuracy of the 1-hidden-layer AE codes); it is read from
     results/task3_classify_ae1/summary/task3_results.json (override with --bottleneck).
     Noise: every input feature is corrupted with probability p (0.2 / 0.4), independently, with a fresh
     corruption pattern in every epoch (see utils/noise.py). The target is always the clean image.
  b) Average reconstruction error on train / validation / test, after training. Reported for
     clean input -> clean target (comparable with Task 2) and for noisy input -> clean target.
  c) One image per class from train / val / test: original, noisy input and reconstructions.
  d) FCNN classification on the DAE codes using the best architecture of Task 3 for that dimension
     (read from the Task 3 results, override with --arch). Validation + test accuracy and confusion matrix.

Training recipe = Task 2 (Adam, lr 1e-3, batch 128, early stopping on validation MSE, patience 5).
Everything runs on the GPU when one is available. Weights are saved for Task 6 as
results/task5_denoising/dae_noise<20|40>_bottleneck_<k>/model.pt (encoder weights: model.encoder[0].weight).
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import get_data_tensors
from models import ARCHITECTURES, build_autoencoder
from train import run_classification_study, default_device
from utils.noise import corrupt, make_generator
from utils.parallel_study import save_codes, make_jobs, pretrain_in_parallel, default_workers
from utils.plotting import save_figure
from utils import training_config as cfg

NOISE_LEVELS = [0.2, 0.4]
DEFAULT_RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "task5_denoising")
TASK2_DIR = os.path.join(CURRENT_DIR, "results", "task2_autoencoders")
TASK3_DIR = os.path.join(CURRENT_DIR, "results", "task3_classify_ae1")


def parse_args():
    p = argparse.ArgumentParser(description="Task 5: denoising autoencoders (20% / 40% noise) + FCNN classification")
    p.add_argument("--noise_levels", type=float, nargs="+", default=NOISE_LEVELS,
                   help="probability that a feature is corrupted in an epoch (default: 0.2 0.4)")
    p.add_argument("--noise_type", choices=["gaussian", "masking"], default="gaussian")
    p.add_argument("--noise_std", type=float, default=0.5, help="std of the Gaussian noise (pixel range is [0, 1])")
    p.add_argument("--no_clip", action="store_true", help="do not clip noisy inputs to [0, 1]")
    p.add_argument("--bottleneck", type=int, default=None,
                   help="bottleneck size (default: best dimension of Task 3, read from its results)")
    p.add_argument("--arch", type=str, default=None,
                   help="FCNN architecture (default: Task 3's best architecture for that dimension)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--task3_dir", type=str, default=TASK3_DIR)
    p.add_argument("--task2_dir", type=str, default=TASK2_DIR)
    # autoencoder training (same recipe as Task 2)
    p.add_argument("--ae_batch_size", type=int, default=cfg.TASK2_BATCH_SIZE)
    p.add_argument("--ae_lr", type=float, default=cfg.TASK2_LEARNING_RATE)
    p.add_argument("--ae_patience", type=int, default=cfg.TASK2_PATIENCE)
    p.add_argument("--ae_min_delta", type=float, default=cfg.TASK2_MIN_DELTA)
    p.add_argument("--ae_max_epochs", type=int, default=1000)
    # classifier training (same recipe as Tasks 1 / 3 / 4)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="classifier batch size (1 = as Task 1)")
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--momentum", type=float, default=cfg.MOMENTUM)
    p.add_argument("--max_epochs", type=int, default=cfg.MAX_EPOCHS)
    p.add_argument("--threshold", type=float, default=cfg.STOPPING_THRESHOLD)
    p.add_argument("--patience", type=int, default=cfg.PATIENCE)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--device", type=str, default=None, help="device for the autoencoders (default: cuda if available)")
    p.add_argument("--clf_device", type=str, default=None, help="device for the classifiers (default: cpu for batch size 1)")
    p.add_argument("--workers", type=int, default=0, help="parallel CPU workers for the classifiers (0 = auto, 1 = sequential)")
    p.add_argument("--no_resume", action="store_true", help="retrain classifiers even if checkpoints exist")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noise_tag(p):
    return f"noise{int(round(p * 100))}"


# ------------------------------------------------------------------ Task 3 hand-over

def best_from_task3(task3_dir, bottleneck=None, arch=None):
    """(bottleneck, arch) = Task 3's best dimension (by test accuracy) and its best architecture."""
    path = os.path.join(task3_dir, "summary", "task3_results.json")
    data = None
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    if bottleneck is None:
        if data is None:
            raise FileNotFoundError(f"{path} not found. Run task3_classify_ae1.py first, "
                                    f"or pass --bottleneck and --arch explicitly.")
        bottleneck = int(data["best_dimension"])
    if arch is None:
        if data is None:
            raise FileNotFoundError(f"{path} not found; pass --arch explicitly.")
        match = [r for r in data["results"] if r["input_dim"] == bottleneck]
        if not match:
            raise ValueError(f"Task 3 has no result for bottleneck {bottleneck}; pass --arch explicitly.")
        arch = match[0]["best_arch"]
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(ARCHITECTURES)}")
    return bottleneck, arch


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ autoencoder training / evaluation

@torch.no_grad()
def batched(model, X, chunk=4096):
    model.eval()
    return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)])


@torch.no_grad()
def mse_between(model, X_in, X_target):
    """Average reconstruction error: mean squared error over all pixels and samples (as in Task 2)."""
    return F.mse_loss(batched(model, X_in), X_target).item()


def train_dae(model, Xtr, Xva_noisy, Xva, p, args, gen):
    """
    Adam on MSE(model(corrupt(x)), x). A fresh corruption pattern is drawn for the whole training set at
    the start of every epoch, so each feature is noisy in a fraction p of the epochs. Early stopping and
    model selection use the validation MSE on one fixed noisy validation set (deterministic).
    """
    dev = Xtr.device
    N = len(Xtr)
    opt = torch.optim.Adam(model.parameters(), lr=args.ae_lr)
    best_val, best_state, stale, history = float("inf"), None, 0, []
    clip = not args.no_clip

    for epoch in range(1, args.ae_max_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=dev, generator=gen)
        Xp = Xtr[perm]
        Xn = corrupt(Xp, p, args.noise_std, args.noise_type, clip, gen)       # new noise every epoch
        total = torch.zeros((), device=dev)
        for i in range(0, N, args.ae_batch_size):
            xb, xn = Xp[i:i + args.ae_batch_size], Xn[i:i + args.ae_batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xn), xb)
            loss.backward()
            opt.step()
            total += loss.detach() * len(xb)
        train_mse = (total / N).item()
        val_mse = mse_between(model, Xva_noisy, Xva)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if epoch <= 3 or epoch % 10 == 0:
            print(f"  epoch {epoch:03d} | train MSE {train_mse:.6f} | validation MSE (noisy in) {val_mse:.6f}", flush=True)

        if val_mse < best_val - args.ae_min_delta:
            best_val, stale = val_mse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.ae_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  stopped after {len(history)} epochs (best validation MSE {best_val:.6f})", flush=True)
    return history


def plot_denoising_grid(original, noisy, recon_noisy, recon_clean, labels, filename, title, idx_to_class=None):
    """First image of every class: original / noisy input / reconstruction from noisy / reconstruction from clean."""
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels).tolist())
    idx = [int(np.flatnonzero(labels == c)[0]) for c in classes]       # same picks as Task 2's grids
    rows = [("Original", original), ("Noisy input", noisy),
            ("Reconstruction\n(from noisy)", recon_noisy), ("Reconstruction\n(from clean)", recon_clean)]
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(2.0 * len(idx) + 1.2, 2.0 * len(rows) + 0.6))
    for r, (name, imgs) in enumerate(rows):
        imgs = imgs.cpu().numpy()
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Class {classes[c]}" + (f" (digit {idx_to_class[classes[c]]})" if idx_to_class else ""), fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(filename)


def save_history(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("epoch", "train_mse", "val_mse"))
        w.writeheader()
        w.writerows(history)


# ------------------------------------------------------------------ summary

def write_summary(ae_results, clf_results, k, arch, task3_ref, task2_ref, args):
    summary_dir = os.path.join(args.results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    L = [f"Task 5 summary: denoising autoencoders (1 hidden layer, bottleneck {k}) + FCNN classification", "=" * 86,
         f"Noise: each feature corrupted with probability p in every epoch; type={args.noise_type}"
         + (f", std={args.noise_std}" if args.noise_type == "gaussian" else "")
         + f", clip to [0,1]={not args.no_clip}", "",
         "Average reconstruction error (MSE vs clean image):",
         f"  {'Model':<22} {'Input':<7} {'Train':>10} {'Val':>10} {'Test':>10}", "  " + "-" * 62]
    if task2_ref:
        L.append(f"  {'Task 2 plain AE':<22} {'clean':<7} {task2_ref['train_mse']:>10.6f} "
                 f"{task2_ref['validation_mse']:>10.6f} {task2_ref['test_mse']:>10.6f}")
    for r in ae_results:
        nm = f"DAE {r['noise_pct']}% noise"
        L.append(f"  {nm:<22} {'clean':<7} {r['clean_in']['train']:>10.6f} {r['clean_in']['val']:>10.6f} {r['clean_in']['test']:>10.6f}")
        L.append(f"  {nm:<22} {'noisy':<7} {r['noisy_in']['train']:>10.6f} {r['noisy_in']['val']:>10.6f} {r['noisy_in']['test']:>10.6f}")
    L += ["", f"FCNN classification on the {k}-dim codes ({arch} {ARCHITECTURES[arch]}, same as Task 3's best for this dimension):",
          f"  {'Representation':<22} {'Val acc':>9} {'Test acc':>9} {'Macro F1':>9} {'vs Task 3':>10}"]
    t3_test = task3_ref["test_acc"] if task3_ref else None
    if task3_ref:
        L.append(f"  {'Task 3 plain AE':<22} {task3_ref['val_acc_by_arch'][arch]:>8.2f}% {task3_ref['test_acc']:>8.2f}% "
                 f"{task3_ref['test_macro_f1']:>8.2f}% {'-':>10}")
    for c in clf_results:
        d = f"{c['test_acc'] - t3_test:+.2f}" if t3_test is not None else "n/a"
        L.append(f"  {'DAE ' + str(c['noise_pct']) + '% noise':<22} {c['val_acc_by_arch'][arch]:>8.2f}% {c['test_acc']:>8.2f}% "
                 f"{c['test_macro_f1']:>8.2f}% {d:>10}")
    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(summary_dir, "task5_summary.txt"), "w") as f:
        f.write(text + "\n")

    with open(os.path.join(summary_dir, "task5_summary.csv"), "w") as f:
        f.write("noise_pct,bottleneck,architecture,epochs,train_mse_clean_in,val_mse_clean_in,test_mse_clean_in,"
                "train_mse_noisy_in,val_mse_noisy_in,test_mse_noisy_in,fcnn_val_acc_pct,fcnn_test_acc_pct,fcnn_test_macro_f1_pct\n")
        for r, c in zip(ae_results, clf_results):
            f.write(f"{r['noise_pct']},{k},{arch},{r['epochs']},{r['clean_in']['train']:.6f},{r['clean_in']['val']:.6f},"
                    f"{r['clean_in']['test']:.6f},{r['noisy_in']['train']:.6f},{r['noisy_in']['val']:.6f},{r['noisy_in']['test']:.6f},"
                    f"{c['val_acc_by_arch'][arch]:.2f},{c['test_acc']:.2f},{c['test_macro_f1']:.2f}\n")

    # test accuracy: plain AE (Task 3) vs the denoising AEs
    labels = (["Task 3\nplain AE"] if task3_ref else []) + [f"DAE {c['noise_pct']}%" for c in clf_results]
    vals = ([task3_ref["test_acc"]] if task3_ref else []) + [c["test_acc"] for c in clf_results]
    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(range(len(vals)), vals, color=["#4C78A8"] + ["#F58518", "#54A24B", "#B279A2"][:len(vals) - 1] if task3_ref
                   else ["#F58518", "#54A24B", "#B279A2"][:len(vals)], width=0.55)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom")
    plt.xticks(range(len(vals)), labels)
    plt.ylim(max(0, min(vals) - 2), 100)
    plt.ylabel("Test accuracy (%)")
    plt.title(f"Task 5: test accuracy on {k}-dim codes ({arch})", fontweight="bold")
    plt.tight_layout()
    save_figure(os.path.join(summary_dir, "test_acc_comparison.png"))

    with open(os.path.join(summary_dir, "task5_results.json"), "w") as f:
        json.dump({"bottleneck": k, "architecture": arch, "noise_type": args.noise_type, "noise_std": args.noise_std,
                   "autoencoders": ae_results, "classification": clf_results,
                   "task3_reference": {"test_acc": t3_test} if task3_ref else None}, f, indent=2)
    print(f"\nSaved results to {args.results_dir}")


# ------------------------------------------------------------------ main

def main():
    args = parse_args()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clf_device = args.clf_device or default_device(args.batch_size)
    k, arch = best_from_task3(args.task3_dir, args.bottleneck, args.arch)
    print(f"AE device: {dev} | classifier device: {clf_device} | bottleneck: {k} | FCNN architecture: {arch} "
          f"{ARCHITECTURES[arch]} | noise levels: {args.noise_levels}", flush=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte), class_to_idx, idx_to_class = get_data_tensors(args.data_dir, device=dev)
    print(f"Loaded: train {tuple(Xtr.shape)}, val {tuple(Xva.shape)}, test {tuple(Xte.shape)}, classes {class_to_idx}", flush=True)
    num_classes = len(idx_to_class)
    clip = not args.no_clip
    os.makedirs(args.results_dir, exist_ok=True)

    ae_results, codes = [], {}
    for p in args.noise_levels:
        tag = noise_tag(p)
        run_dir = os.path.join(args.results_dir, f"dae_{tag}_bottleneck_{k}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== Denoising AE: {p * 100:.0f}% noise, bottleneck {k} ===", flush=True)

        set_seed(args.seed)                                        # each noise level reproducible on its own
        gen = make_generator(dev, args.seed)
        model = build_autoencoder("ae1", k).to(dev)                # 784 -> k (linear) -> 784 (sigmoid)

        # fixed noisy copies of every split, used for early stopping and for the "noisy input" errors
        eval_gen = make_generator(dev, args.seed + 12345)
        noisy = {s: corrupt(X, p, args.noise_std, args.noise_type, clip, eval_gen)
                 for s, X in (("train", Xtr), ("val", Xva), ("test", Xte))}
        clean = {"train": Xtr, "val": Xva, "test": Xte}

        t0 = time.time()
        history = train_dae(model, Xtr, noisy["val"], Xva, p, args, gen)
        elapsed = time.time() - t0

        # (b) average reconstruction errors after training
        clean_in = {s: mse_between(model, clean[s], clean[s]) for s in clean}
        noisy_in = {s: mse_between(model, noisy[s], clean[s]) for s in clean}
        print(f"  reconstruction MSE, clean input : " + " | ".join(f"{s} {v:.6f}" for s, v in clean_in.items()))
        print(f"  reconstruction MSE, noisy input : " + " | ".join(f"{s} {v:.6f}" for s, v in noisy_in.items()), flush=True)

        # (c) reconstructed images, one per class, for train / val / test
        labels = {"train": ytr, "val": yva, "test": yte}
        for s, nm in (("train", "train"), ("val", "validation"), ("test", "test")):
            plot_denoising_grid(clean[s].cpu(), noisy[s].cpu(), batched(model, noisy[s]).cpu(),
                                batched(model, clean[s]).cpu(), labels[s].cpu(),
                                os.path.join(run_dir, f"{s}_reconstructions.png"),
                                f"Denoising AE, {p * 100:.0f}% noise, bottleneck {k} - {nm}", idx_to_class)

        torch.save({kk: v.cpu() for kk, v in model.state_dict().items()}, os.path.join(run_dir, "model.pt"))
        save_history(history, os.path.join(run_dir, "loss_history.csv"))

        # (d) compressed representation of clean train / val / test data -> saved for the classifier
        model.eval()
        with torch.no_grad():
            Z = {s: torch.cat([model.encode(clean[s][i:i + 4096]) for i in range(0, len(clean[s]), 4096)]).float()
                 for s in clean}
        clf_dir = os.path.join(args.results_dir, f"classifier_{tag}_bottleneck_{k}")
        codes_path = os.path.join(clf_dir, "codes.npz")
        save_codes(codes_path, Z["train"], ytr, Z["val"], yva, Z["test"], yte)
        codes[p] = (Z, clf_dir, codes_path)

        ae_results.append({"noise": p, "noise_pct": int(round(p * 100)), "bottleneck": k, "epochs": len(history),
                           "train_time_s": round(elapsed, 1), "clean_in": clean_in, "noisy_in": noisy_in})
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- (d) FCNN classification: the best Task 3 architecture, one per noise level ----
    hyper = dict(batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, max_epochs=args.max_epochs,
                 threshold=args.threshold, patience=args.patience, seed=args.seed, activation=cfg.ACTIVATION)
    workers = args.workers or default_workers(len(args.noise_levels))
    study_resume = not args.no_resume
    if clf_device == "cpu" and workers > 1 and len(args.noise_levels) > 1:
        jobs = []
        for p in args.noise_levels:
            jobs += make_jobs(codes[p][2], codes[p][1], [arch], num_classes, hyper)
        pretrain_in_parallel(jobs, workers, resume=not args.no_resume)
        study_resume = True

    clf_results = []
    for p in args.noise_levels:
        Z, clf_dir, _ = codes[p]
        res = run_classification_study(
            name=f"DAE{int(round(p * 100))}-{k}", X_train=Z["train"].to(clf_device), y_train=ytr.to(clf_device),
            X_val=Z["val"].to(clf_device), y_val=yva.to(clf_device), X_test=Z["test"].to(clf_device),
            y_test=yte.to(clf_device), idx_to_class=idx_to_class, out_dir=clf_dir, archs=[arch],
            batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, max_epochs=args.max_epochs,
            threshold=args.threshold, patience=args.patience, seed=args.seed, resume=study_resume)
        res["noise_pct"] = int(round(p * 100))
        clf_results.append(res)

    # references for the comparison (both optional)
    t3 = load_json(os.path.join(args.task3_dir, "summary", "task3_results.json"))
    task3_ref = next((r for r in t3["results"] if r["input_dim"] == k), None) if t3 else None
    if task3_ref is not None and task3_ref["best_arch"] != arch:
        task3_ref = None                                       # only compare like with like
    t2 = load_json(os.path.join(args.task2_dir, "summary.json"))
    task2_ref = next((r for r in t2 if r["architecture"] == "ae1" and r["bottleneck"] == k), None) if t2 else None

    write_summary(ae_results, clf_results, k, arch, task3_ref, task2_ref, args)
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(ae_results, f, indent=2)


if __name__ == "__main__":
    main()
