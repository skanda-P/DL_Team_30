# Runs the independent (representation, architecture) FCNN trainings of a classification study in parallel CPU processes.
"""
Speed-up helper for Tasks 3 and 5.

The recipe inherited from Assignment 3 / Task 1 is SGD with batch size 1. That is latency-bound
(tiny matrices, one optimiser step per sample), so a GPU does not help and a single CPU core is the
bottleneck. But every (representation, architecture) training is independent of all the others, so we
simply run them in separate worker processes (one core each).

Each worker trains one FCNN with the *same* functions from train.py and writes exactly the checkpoint
files that train.run_classification_study() writes (runs/<arch>/run.json + model.pt, same config
dict). run_classification_study(..., resume=True) afterwards finds every checkpoint, skips training and
only does the selection / test evaluation / plots. Results are therefore identical in format and
(up to floating-point noise) in value to the plain sequential run.
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def save_codes(path, Ztr, ytr, Zva, yva, Zte, yte):
    """Persist compressed representations (and labels) of all three splits."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path,
             Xtr=Ztr.cpu().numpy().astype(np.float32), ytr=ytr.cpu().numpy(),
             Xva=Zva.cpu().numpy().astype(np.float32), yva=yva.cpu().numpy(),
             Xte=Zte.cpu().numpy().astype(np.float32), yte=yte.cpu().numpy())


def default_workers(n_jobs):
    return max(1, min(int(n_jobs), os.cpu_count() or 1))


def make_jobs(codes_path, out_dir, archs, num_classes, hyper):
    """One job per architecture. `hyper` holds batch_size, lr, momentum, max_epochs, threshold,
    patience, seed, activation (the same values later passed to run_classification_study)."""
    return [{"codes_path": codes_path, "run_dir": os.path.join(out_dir, "runs", a), "arch": a,
             "num_classes": num_classes, **hyper} for a in archs]


def _train_job(job):
    """Worker: train + checkpoint one architecture. Mirrors run_classification_study's inner loop."""
    torch.set_num_threads(1)
    from models import ARCHITECTURES, build_classifier, count_parameters
    from train import train_classifier, evaluate, _run_config, _to_py

    arch, run_dir = job["arch"], job["run_dir"]
    json_path, model_path = os.path.join(run_dir, "run.json"), os.path.join(run_dir, "model.pt")

    d = np.load(job["codes_path"])
    Xtr, ytr = torch.from_numpy(d["Xtr"]).float(), torch.from_numpy(d["ytr"]).long()
    Xva, yva = torch.from_numpy(d["Xva"]).float(), torch.from_numpy(d["yva"]).long()
    input_dim = Xtr.shape[1]

    run_cfg = _run_config(input_dim, job["batch_size"], job["lr"], job["momentum"], job["threshold"],
                          job["patience"], job["seed"], job["activation"])

    if job["resume"] and os.path.exists(json_path) and os.path.exists(model_path):
        with open(json_path) as f:
            if json.load(f).get("config") == run_cfg:
                return f"[{os.path.basename(os.path.dirname(os.path.dirname(run_dir)))}/{arch}] already trained, skipped"

    model = build_classifier(arch, input_dim, job["num_classes"], job["activation"], job["seed"])
    res = train_classifier(model, Xtr, ytr, job["batch_size"], job["lr"], job["momentum"], job["max_epochs"],
                           job["threshold"], job["patience"], job["seed"], verbose=False, tag=arch)
    tr_loss, tr_acc, _ = evaluate(model, Xtr, ytr)
    va_loss, va_acc, _ = evaluate(model, Xva, yva)
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
    tag = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    return (f"[{tag}/{arch}] epochs={record['epochs']} ({record['elapsed_s']:.0f}s) | "
            f"train acc {tr_acc*100:.2f}% | val acc {va_acc*100:.2f}%")


def pretrain_in_parallel(jobs, workers, resume=True):
    """Train every job in a pool of `workers` processes; prints one line per finished job."""
    jobs = [{**j, "resume": resume} for j in jobs]
    # biggest networks first -> better load balancing at the end of the pool
    jobs.sort(key=lambda j: (len(j["arch"]), j["arch"]), reverse=True)
    workers = max(1, min(workers, len(jobs)))
    print(f"Training {len(jobs)} classifiers in parallel on {workers} CPU worker process(es) ...", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(_train_job, j) for j in jobs]
        for fut in as_completed(futures):
            done += 1
            print(f"  ({done}/{len(jobs)}, {time.time() - t0:.0f}s) {fut.result()}", flush=True)
    print(f"Parallel training finished in {time.time() - t0:.0f}s.", flush=True)
