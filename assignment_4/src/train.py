# Shared training and evaluation routines for autoencoders and FCNN classifiers, including loss computation and single-run CLI support.
"""
Training / evaluation utilities for the FCNN classifier (Tasks 1, 3, 4, 5).

run_classification_study() is the shared "Step b" of Tasks 1, 3 and 4:
  for ONE fixed input representation (PCA features, 1-hidden-layer AE codes, ...)
  it trains every FCNN architecture, reports validation accuracy for each,
  selects the best architecture by VALIDATION accuracy, and only then
  evaluates that single model on the test set (accuracy + confusion matrix).

Training recipe = best recipe from Assignment 3: cross-entropy, SGD + momentum,
batch size 1, tanh, Xavier-normal init, stop when |L_t - L_{t-1}| < 1e-4.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models import ARCHITECTURES, ARCH_GROUPS, build_classifier, count_parameters
from utils.metrics import classification_metrics
from utils.plotting import plot_error_vs_epochs, plot_confusion_matrix_heatmap
from utils import training_config as cfg


# --------------------------------------------------------------------- helpers

def default_device(batch_size):
    """Batch size 1 on tiny MLPs is faster on CPU (GPU kernel-launch overhead dominates)."""
    if batch_size == 1:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, X, y, chunk=4096):
    """Returns (avg cross-entropy loss, accuracy in [0,1], predictions as numpy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds = 0.0, []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            logits = model(X[i:i + chunk])
            total_loss += criterion(logits, y[i:i + chunk]).item()
            preds.append(torch.argmax(logits, dim=1))
    preds = torch.cat(preds).cpu().numpy()
    acc = float((preds == y.cpu().numpy()).mean())
    return total_loss / len(X), acc, preds


def train_classifier(model, X_train, y_train, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
                     momentum=cfg.MOMENTUM, max_epochs=cfg.MAX_EPOCHS,
                     threshold=cfg.STOPPING_THRESHOLD, patience=cfg.PATIENCE,
                     seed=cfg.SEED, verbose=True, tag=""):
    """
    Trains `model` in place. The convergence rule is the one used in Assignment 3:
    stop once |L_t - L_{t-1}| < threshold for `patience` consecutive epochs, where
    L_t is the average training loss accumulated over epoch t.
    batch_size=None means full-batch.
    """
    device = X_train.device
    N = len(X_train)
    bs = N if batch_size in (None, 0) else int(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    gen = torch.Generator().manual_seed(seed)      # reproducible shuffling

    initial_loss, _, _ = evaluate(model, X_train, y_train)
    epoch_losses, consecutive, converged = [], 0, False
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(N, generator=gen).to(device)
        Xp, yp = X_train[perm], y_train[perm]

        running = torch.zeros(1, device=device)
        for i in range(0, N, bs):
            xb, yb = Xp[i:i + bs], yp[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.detach() * len(xb)
        avg_loss = (running / N).item()
        epoch_losses.append(avg_loss)

        diff = abs(epoch_losses[-1] - epoch_losses[-2]) if epoch > 1 else float("inf")
        if diff < threshold:
            consecutive += 1
            if consecutive >= patience:
                converged = True
        else:
            consecutive = 0

        if verbose and (epoch <= 3 or epoch % 5 == 0 or converged):
            d = f"{diff:.6f}" if diff != float("inf") else "N/A"
            print(f"    {tag} epoch {epoch:4d} | loss {avg_loss:.6f} | |diff| {d}", flush=True)
        if converged:
            break

    return {"epoch_losses": epoch_losses, "initial_loss": initial_loss,
            "converged": converged, "elapsed": time.time() - start}


def _to_py(obj):
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _run_config(input_dim, batch_size, lr, momentum, threshold, patience, seed, activation):
    return {"input_dim": input_dim, "batch_size": batch_size, "lr": lr, "momentum": momentum,
            "threshold": threshold, "patience": patience, "seed": seed, "activation": activation}


# ------------------------------------------------------------- the study runner

def run_classification_study(name, X_train, y_train, X_val, y_val, X_test, y_test, idx_to_class,
                             out_dir, archs=None, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
                             momentum=cfg.MOMENTUM, max_epochs=cfg.MAX_EPOCHS,
                             threshold=cfg.STOPPING_THRESHOLD, patience=cfg.PATIENCE,
                             activation=cfg.ACTIVATION, seed=cfg.SEED, resume=True, verbose=True):
    """
    Train every architecture on (X_train, y_train); report validation accuracy;
    select the best architecture by validation accuracy; evaluate ONLY that model on test.

    All X_* are float tensors of shape (n, input_dim) on the same device; y_* are long tensors.
    Each trained model is checkpointed in <out_dir>/runs/<arch>/ so an interrupted study
    can be resumed (resume=True) without retraining finished architectures.
    """
    archs = archs or ARCH_GROUPS["all"]
    input_dim = X_train.shape[1]
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    device = X_train.device
    os.makedirs(out_dir, exist_ok=True)
    run_cfg = _run_config(input_dim, batch_size, lr, momentum, threshold, patience, seed, activation)

    print(f"\n=== {name}: input_dim={input_dim}, {len(archs)} architectures ===", flush=True)
    rows, models = [], {}

    for arch in archs:
        run_dir = os.path.join(out_dir, "runs", arch)
        json_path, model_path = os.path.join(run_dir, "run.json"), os.path.join(run_dir, "model.pt")
        model = build_classifier(arch, input_dim, num_classes, activation, seed).to(device)

        record = None
        if resume and os.path.exists(json_path) and os.path.exists(model_path):
            with open(json_path) as f:
                saved = json.load(f)
            if saved.get("config") == run_cfg:
                model.load_state_dict(torch.load(model_path, map_location=device))
                record = saved
                print(f"  [{arch}] resumed from checkpoint (val acc {record['val_acc']*100:.2f}%)", flush=True)

        if record is None:
            print(f"  [{arch}] {ARCHITECTURES[arch]}  params={count_parameters(model):,}", flush=True)
            res = train_classifier(model, X_train, y_train, batch_size, lr, momentum, max_epochs,
                                   threshold, patience, seed, verbose, tag=arch)
            tr_loss, tr_acc, _ = evaluate(model, X_train, y_train)
            va_loss, va_acc, _ = evaluate(model, X_val, y_val)
            record = {
                "arch": arch, "hidden_dims": ARCHITECTURES[arch], "params": count_parameters(model),
                "epochs": len(res["epoch_losses"]), "converged": res["converged"],
                "elapsed_s": round(res["elapsed"], 2), "initial_loss": res["initial_loss"],
                "epoch_losses": res["epoch_losses"],
                "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                "config": run_cfg,
            }
            os.makedirs(run_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            with open(json_path, "w") as f:
                json.dump(_to_py(record), f)
            print(f"  [{arch}] epochs={record['epochs']} ({record['elapsed_s']:.0f}s) | "
                  f"train acc {tr_acc*100:.2f}% | val acc {va_acc*100:.2f}%", flush=True)

        rows.append(record)
        models[arch] = model

    # ---- validation table (Step b.ii) ----
    rows_sorted = sorted(rows, key=lambda r: (-r["val_acc"], r["val_loss"]))
    best = rows_sorted[0]               # best by val accuracy; ties broken by lower val loss
    best_arch, best_model = best["arch"], models[best["arch"]]

    csv_path = os.path.join(out_dir, "validation_accuracy.csv")
    with open(csv_path, "w") as f:
        f.write("architecture,hidden_layers,parameters,epochs,train_acc_pct,val_acc_pct,val_loss\n")
        for r in rows:
            f.write(f"{r['arch']},\"{'-'.join(map(str, r['hidden_dims']))}\",{r['params']},{r['epochs']},"
                    f"{r['train_acc']*100:.2f},{r['val_acc']*100:.2f},{r['val_loss']:.5f}\n")

    # ---- test evaluation of the selected architecture only (Step b.iii) ----
    _, test_acc, test_preds = evaluate(best_model, X_test, y_test)
    tm = classification_metrics(y_test.cpu().numpy(), test_preds, num_classes)
    cm = tm["confusion_matrix"]

    np.savetxt(os.path.join(out_dir, "test_confusion_matrix.csv"), cm, fmt="%d", delimiter=",",
               header="rows=true class, cols=predicted class; order: " + ",".join(class_names))
    plot_confusion_matrix_heatmap(
        cm, class_names, title=f"Test confusion matrix - {name}\n{best_arch}, test acc {test_acc*100:.2f}%",
        filename=os.path.join(out_dir, "test_confusion_matrix.png"))
    plot_error_vs_epochs(
        best["epoch_losses"], title=f"Training loss - {name} ({best_arch})",
        filename=os.path.join(out_dir, "best_training_loss.png"), initial_value=best["initial_loss"])

    lines = [
        f"Representation:          {name} (input_dim = {input_dim})",
        f"Selected architecture:   {best_arch}  {ARCHITECTURES[best_arch]}   (selected by validation accuracy)",
        f"Validation accuracy:     {best['val_acc']*100:.2f}%",
        f"Test accuracy:           {test_acc*100:.2f}%",
        f"Test macro precision:    {tm['macro_precision']*100:.2f}%",
        f"Test macro recall:       {tm['macro_recall']*100:.2f}%",
        f"Test macro F1:           {tm['macro_f_measure']*100:.2f}%",
        f"Epochs to converge:      {best['epochs']}",
        "", "Validation accuracy of all architectures (best first):",
    ]
    for r in rows_sorted:
        lines.append(f"  {r['arch']:11s} {'-'.join(map(str, r['hidden_dims'])):22s} "
                     f"val acc {r['val_acc']*100:6.2f}%   epochs {r['epochs']:4d}")
    lines += ["", f"Test confusion matrix (rows = true, cols = predicted; classes {class_names}):", str(cm), "",
              "Per-class test metrics:"]
    for i, c in enumerate(class_names):
        lines.append(f"  digit {c}: precision {tm['class_precision'][i]:.4f}  "
                     f"recall {tm['class_recall'][i]:.4f}  F1 {tm['class_f_measure'][i]:.4f}")
    summary_text = "\n".join(lines)
    with open(os.path.join(out_dir, "best_model_report.txt"), "w") as f:
        f.write(summary_text + "\n")
    print("\n" + summary_text + "\n", flush=True)

    result = {
        "name": name, "input_dim": input_dim, "best_arch": best_arch,
        "best_hidden_dims": ARCHITECTURES[best_arch], "best_val_acc": best["val_acc"] * 100,
        "test_acc": test_acc * 100, "test_macro_f1": tm["macro_f_measure"] * 100,
        "test_macro_precision": tm["macro_precision"] * 100, "test_macro_recall": tm["macro_recall"] * 100,
        "confusion_matrix": cm.tolist(), "class_names": class_names,
        "val_acc_by_arch": {r["arch"]: r["val_acc"] * 100 for r in rows},
        "epochs_by_arch": {r["arch"]: r["epochs"] for r in rows},
    }
    with open(os.path.join(out_dir, "study_result.json"), "w") as f:
        json.dump(_to_py(result), f, indent=2)
    return result