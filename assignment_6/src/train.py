"""Single-configuration RNN/LSTM training CLI and training loop for Assignment 6."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from data_loader import (
    DATA_ROOT,
    DEFAULT_SPLIT_SEED,
    RESULTS_ROOT,
    PreparedData,
    prepare_dataloaders,
)
from models import SeqClassifier
from utils.convergence import ConvergenceChecker
from utils.io import compute_run_id, load_json, save_json, setup_logger
from utils.metrics import compute_classification_metrics, evaluate_model
from utils.seed import set_seed

MODELS_DIR = Path(__file__).resolve().parent / "models"
logger = setup_logger("train")


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def build_default_config(
    dataset: str,
    model: str,
    hidden_size: int = 64,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.0,
    optimizer: str = "adam",
    lr: float = 1e-3,
    batch_size: int = 32,
    grad_clip: Optional[float] = 1.0,
    tol: float = 1e-4,
    max_epochs: int = 1000,
    seed: int = 42,
    split_seed: int = DEFAULT_SPLIT_SEED,
    stage: str = "single",
) -> Dict[str, Any]:
    """Build a canonical hyperparameter config dictionary."""
    norm_clip = None if (grad_clip is None or float(grad_clip) <= 0.0) else float(grad_clip)
    return {
        "dataset": dataset,
        "model": model,
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bidirectional": bool(bidirectional),
        "dropout": float(dropout),
        "optimizer": optimizer,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "grad_clip": norm_clip,
        "tol": float(tol),
        "max_epochs": int(max_epochs),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "stage": stage,
    }


def save_checkpoint(
    path: Path,
    state_dict: Dict[str, Any],
    config: Dict[str, Any],
    prepared_data: PreparedData,
    param_count: int,
) -> None:
    """Persist model weights and preprocessing metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": config,
            "input_size": prepared_data.input_size,
            "class_names": prepared_data.class_names,
            "param_count": param_count,
            "scaler_mean": prepared_data.scaler_mean,
            "scaler_std": prepared_data.scaler_std,
        },
        path,
    )


def run_training(
    config: Dict[str, Any],
    data_root: str | Path = DATA_ROOT,
    results_root: str | Path = RESULTS_ROOT,
    models_dir: str | Path = MODELS_DIR,
    prepared_data: Optional[PreparedData] = None,
    keep: bool = False,
    force: bool = False,
    device_arg: str = "auto",
    return_state: bool = False,
) -> Dict[str, Any]:
    """Train one configuration until successive-epoch training error diff < 1e-4."""
    run_id = compute_run_id(config)
    run_dir = Path(results_root) / config["dataset"] / config["model"] / config["stage"] / run_id
    metrics_path = run_dir / "metrics.json"

    if metrics_path.is_file() and not force:
        existing = load_json(metrics_path)
        existing["run_id"] = run_id
        existing["run_dir"] = str(run_dir)
        return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else torch.device(device_arg)

    if prepared_data is None:
        prepared_data = prepare_dataloaders(
            dataset=config["dataset"],
            data_root=data_root,
            results_root=results_root,
            batch_size=int(config["batch_size"]),
            split_seed=int(config["split_seed"]),
            seed=int(config["seed"]),
        )

    model = SeqClassifier(
        rnn_type=config["model"],
        input_size=prepared_data.input_size,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        bidirectional=bool(config["bidirectional"]),
        dropout=float(config["dropout"]),
        num_classes=len(prepared_data.class_names),
    ).to(device)

    param_count = model.count_parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(config["optimizer"], model.parameters(), lr=float(config["lr"]))
    checker = ConvergenceChecker(tol=float(config["tol"]), min_epochs=2, max_epochs=int(config["max_epochs"]))

    train_eval_batches = [(bx.to(device), bl, by.to(device)) for bx, bl, by in prepared_data.train_eval_loader]
    val_batches = [(bx.to(device), bl, by.to(device)) for bx, bl, by in prepared_data.val_loader]
    test_batches = [(bx.to(device), bl, by.to(device)) for bx, bl, by in prepared_data.test_loader]

    train_seqs_dev = [s.to(device) for s in prepared_data.train_loader.dataset.sequences]
    train_lens = [int(s.size(0)) for s in train_seqs_dev]
    train_labels_dev = torch.tensor(prepared_data.train_loader.dataset.labels, dtype=torch.int64, device=device)
    num_train = len(train_seqs_dev)
    feat_dim = int(prepared_data.input_size)
    batch_size = int(config["batch_size"])
    shuffle_gen = torch.Generator().manual_seed(int(config["seed"]))

    full_config = {
        **config,
        "run_id": run_id,
        "input_size": prepared_data.input_size,
        "num_classes": len(prepared_data.class_names),
        "class_names": prepared_data.class_names,
        "param_count": param_count,
    }
    save_json(full_config, run_dir / "config.json")

    history = []
    diverged = False
    stop_reason = "unknown"
    converged = False
    train_start = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        ep_start = time.perf_counter()
        model.train()

        perm = torch.randperm(num_train, generator=shuffle_gen).tolist()
        for start_i in range(0, num_train, batch_size):
            idx = perm[start_i : start_i + batch_size]
            bl = torch.tensor([train_lens[k] for k in idx], dtype=torch.int64)
            tmax = int(bl.max().item())
            bx = torch.zeros((len(idx), tmax, feat_dim), dtype=torch.float32, device=device)
            for j, k in enumerate(idx):
                bx[j, : train_lens[k], :] = train_seqs_dev[k]
            by = train_labels_dev[idx]

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx, bl), by)
            if not torch.isfinite(loss):
                diverged = True
                stop_reason = "diverged"
                break

            loss.backward()
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["grad_clip"]))
            optimizer.step()

        if diverged:
            break

        train_eval = evaluate_model(model, train_eval_batches, criterion, device)
        val_eval = evaluate_model(model, val_batches, criterion, device)
        train_error = float(train_eval["loss"])

        history.append(
            {
                "epoch": epoch,
                "train_error": train_error,
                "train_acc": float(train_eval["accuracy"]),
                "val_loss": float(val_eval["loss"]),
                "val_acc": float(val_eval["accuracy"]),
                "epoch_time_sec": float(time.perf_counter() - ep_start),
            }
        )

        should_stop, reason = checker.step(epoch, train_error)
        if should_stop:
            stop_reason = reason or "unknown"
            converged = checker.converged
            diverged = stop_reason == "diverged"
            break

    total_train_time = time.perf_counter() - train_start
    save_json(history, run_dir / "history.json")

    if diverged:
        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "stage": config["stage"],
            "config": config,
            "class_names": prepared_data.class_names,
            "param_count": param_count,
            "epochs": len(history),
            "converged": False,
            "diverged": True,
            "stop_reason": stop_reason,
            "train_error": float("nan"),
            "train_acc": 0.0,
            "val_loss": float("inf"),
            "val_acc": 0.0,
            "test_loss": float("inf"),
            "test_acc": 0.0,
            "total_train_time_sec": float(total_train_time),
        }
        save_json(metrics, metrics_path)
        return metrics

    final_train = evaluate_model(model, train_eval_batches, criterion, device)
    final_val = evaluate_model(model, val_batches, criterion, device)
    final_test = evaluate_model(model, test_batches, criterion, device)

    train_cm_info = compute_classification_metrics(
        final_train["y_true"], final_train["y_pred"], prepared_data.class_names
    )
    test_cm_info = compute_classification_metrics(
        final_test["y_true"], final_test["y_pred"], prepared_data.class_names
    )

    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()} if (keep or return_state) else None
    checkpoint_path_str: Optional[str] = None
    if keep and state_cpu is not None:
        ckpt_path = Path(models_dir) / f"{config['dataset']}_{config['model']}_{run_id}.pt"
        save_checkpoint(ckpt_path, state_cpu, full_config, prepared_data, param_count)
        checkpoint_path_str = str(ckpt_path)

    metrics = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage": config["stage"],
        "config": config,
        "class_names": prepared_data.class_names,
        "param_count": param_count,
        "epochs": len(history),
        "converged": bool(converged),
        "diverged": False,
        "stop_reason": stop_reason,
        "train_error": float(final_train["loss"]),
        "train_acc": float(final_train["accuracy"]),
        "val_loss": float(final_val["loss"]),
        "val_acc": float(final_val["accuracy"]),
        "test_loss": float(final_test["loss"]),
        "test_acc": float(final_test["accuracy"]),
        "total_train_time_sec": float(total_train_time),
        "train_cm": train_cm_info["confusion_matrix"],
        "test_cm": test_cm_info["confusion_matrix"],
        "test_cm_normalized": test_cm_info["confusion_matrix_normalized"],
        "test_classification_report_text": test_cm_info["classification_report_text"],
        "checkpoint_path": checkpoint_path_str,
    }
    save_json(metrics, metrics_path)
    if return_state and state_cpu is not None:
        metrics["_state_dict"] = state_cpu
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single RNN/LSTM configuration.")
    parser.add_argument("--dataset", type=str, required=True, choices=["handwritten", "cv"])
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--stage", type=str, default="single")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    args = parser.parse_args()

    cfg = build_default_config(
        dataset=args.dataset,
        model=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        optimizer=args.optimizer,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        tol=args.tol,
        max_epochs=args.max_epochs,
        seed=args.seed,
        split_seed=args.split_seed,
        stage=args.stage,
    )
    res = run_training(
        cfg,
        data_root=args.data_root,
        results_root=args.results_root,
        keep=args.keep,
        force=args.force,
        device_arg=args.device,
    )
    logger.info(
        "%s/%s [%s] epochs=%d stop=%s train_acc=%.4f val_acc=%.4f test_acc=%.4f time=%.2fs",
        args.dataset,
        args.model,
        res["run_id"],
        res["epochs"],
        res["stop_reason"],
        res["train_acc"],
        res["val_acc"],
        res["test_acc"],
        res["total_train_time_sec"],
    )


if __name__ == "__main__":
    main()
