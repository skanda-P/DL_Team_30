from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
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
from utils.io import compute_run_id, load_json, save_csv_and_md, save_json, setup_logger
from utils.metrics import compute_classification_metrics, evaluate_model
from utils.plotting import generate_all_plots_for_dataset, update_summary_markdown
from utils.seed import set_seed

DATASET_NAME = "handwritten"
MODELS_DIR = Path(__file__).resolve().parent / "models"
logger = setup_logger("part_1")

STAGE1_HIDDEN_SIZES = [32, 64, 128]
STAGE1_NUM_LAYERS = [1, 2, 3]
STAGE2_ALT_LRS = [3e-3, 3e-4]
STAGE2_SGD_LR = 1e-2
STAGE2_DROPOUT = 0.2


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def build_default_config(
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
    max_epochs: int = 3000,
    seed: int = 42,
    split_seed: int = DEFAULT_SPLIT_SEED,
    stage: str = "single",
) -> Dict[str, Any]:
    norm_clip = None if (grad_clip is None or float(grad_clip) <= 0.0) else float(grad_clip)
    return {
        "dataset": DATASET_NAME,
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


def select_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    converged = [r for r in results if r.get("converged") and not r.get("diverged")]
    candidates = converged if converged else results
    if not converged:
        logger.warning("No model converged before reaching max_epochs limit. Selecting from non-converged runs.")
    return max(
        candidates,
        key=lambda r: (
            1 if (r.get("converged") and not r.get("diverged")) else 0,
            float(r["val_acc"]),
            -float(r["val_loss"]) if np.isfinite(r["val_loss"]) else -1e12,
            -int(r["param_count"]),
        ),
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
    full_config = build_default_config(
        model=config["model"],
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 1),
        bidirectional=config.get("bidirectional", False),
        dropout=config.get("dropout", 0.0),
        optimizer=config.get("optimizer", "adam"),
        lr=config.get("lr", 1e-3),
        batch_size=config.get("batch_size", 32),
        grad_clip=config.get("grad_clip", 1.0),
        tol=config.get("tol", 1e-4),
        max_epochs=config.get("max_epochs", 3000),
        seed=config.get("seed", 42),
        split_seed=config.get("split_seed", DEFAULT_SPLIT_SEED),
        stage=config.get("stage", "single"),
    )
    config = full_config

    run_id = compute_run_id(config)
    run_dir = Path(results_root) / DATASET_NAME / config["model"] / config["stage"] / run_id
    metrics_path = run_dir / "metrics.json"

    if metrics_path.is_file() and not force:
        saved = load_json(metrics_path)
        if return_state:
            ckpt_p = Path(models_dir) / f"{DATASET_NAME}_{config['model']}_best.pt"
            if ckpt_p.is_file():
                try:
                    loaded = torch.load(ckpt_p, map_location="cpu")
                    saved["_state_dict"] = loaded.get("model_state_dict")
                except Exception:
                    pass
        return saved

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))

    if prepared_data is None:
        prepared_data = prepare_dataloaders(
            dataset=DATASET_NAME,
            data_root=data_root,
            results_root=results_root,
            batch_size=int(config["batch_size"]),
            split_seed=int(config["split_seed"]),
            seed=int(config["seed"]),
        )

    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SeqClassifier(
        rnn_type=config["model"],
        input_size=prepared_data.input_size,
        hidden_size=int(config["hidden_size"]),
        num_classes=len(prepared_data.class_names),
        num_layers=int(config["num_layers"]),
        bidirectional=bool(config["bidirectional"]),
        dropout=float(config["dropout"]),
    ).to(device)

    param_count = model.count_parameters()
    optimizer = _build_optimizer(config["optimizer"], model.parameters(), lr=float(config["lr"]))
    criterion = nn.CrossEntropyLoss()
    checker = ConvergenceChecker(tol=float(config["tol"]), min_epochs=2, max_epochs=int(config["max_epochs"]))

    train_eval_batches = [(bx.to(device), bl, by.to(device)) for bx, bl, by in prepared_data.train_eval_loader]
    val_batches = [(bx.to(device), bl, by.to(device)) for bx, bl, by in prepared_data.val_loader]
    test_batches = [(bx.to(device), bl, by.to(device)) for bx, bl, by in prepared_data.test_loader]

    train_seqs_dev = [s.to(device) for s in prepared_data.train_loader.dataset.sequences]
    train_lens = [int(s.size(0)) for s in train_seqs_dev]
    train_labels_dev = torch.tensor(prepared_data.train_loader.dataset.labels, dtype=torch.int64, device=device)
    num_train = len(train_seqs_dev)
    batch_size = int(config["batch_size"])
    feat_dim = int(prepared_data.input_size)

    history: List[Dict[str, Any]] = []
    converged = False
    diverged = False
    stop_reason = "max_epochs_cap"
    shuffle_gen = torch.Generator(device="cpu").manual_seed(int(config["seed"]))

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
        ckpt_path = Path(models_dir) / f"{DATASET_NAME}_{config['model']}_{run_id}.pt"
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


def run_sweep(
    model_type: str,
    prep: PreparedData,
    seed: int = 42,
    max_epochs: int = 3000,
    tol: float = 1e-4,
    split_seed: int = DEFAULT_SPLIT_SEED,
    data_root: str | Path = DATA_ROOT,
    results_root: str | Path = RESULTS_ROOT,
    models_dir: str | Path = MODELS_DIR,
    force: bool = False,
    device_arg: str = "auto",
) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    states_by_run: Dict[str, Any] = {}

    def _execute(cfgs: List[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
        batch_res = []
        for cfg in cfgs:
            res = run_training(
                {**cfg, "stage": stage},
                data_root=data_root,
                results_root=results_root,
                models_dir=models_dir,
                prepared_data=prep,
                keep=False,
                force=force,
                device_arg=device_arg,
                return_state=True,
            )
            state = res.pop("_state_dict", None)
            if state is not None:
                states_by_run[res["run_id"]] = state
            batch_res.append(res)
            all_results.append(res)
            logger.info(
                "[%s/%s | %-11s] h=%3d l=%d bi=%s do=%.1f opt=%-7s lr=%-6g | ep=%3d (%s) | "
                "train=%.2f%% val=%.2f%% test=%.2f%% (%.1fs)",
                DATASET_NAME,
                model_type,
                stage,
                cfg["hidden_size"],
                cfg["num_layers"],
                "Y" if cfg["bidirectional"] else "N",
                cfg["dropout"],
                cfg["optimizer"],
                cfg["lr"],
                res["epochs"],
                res["stop_reason"],
                res["train_acc"] * 100.0,
                res["val_acc"] * 100.0,
                res["test_acc"] * 100.0,
                res["total_train_time_sec"],
            )
        return batch_res

    logger.info("Stage 1 sweep (%s / %s): 9 architectures", DATASET_NAME, model_type)
    s1_cfgs = [
        build_default_config(
            model=model_type,
            hidden_size=hs,
            num_layers=nl,
            tol=tol,
            max_epochs=max_epochs,
            seed=seed,
            split_seed=split_seed,
        )
        for nl in STAGE1_NUM_LAYERS
        for hs in STAGE1_HIDDEN_SIZES
    ]
    s1_results = _execute(s1_cfgs, "stage1_arch")
    best_s1 = select_best(s1_results)

    logger.info("Stage 2 tuning (%s / %s): 6 configurations", DATASET_NAME, model_type)
    s2_lr_results = _execute([{**best_s1["config"], "lr": lr} for lr in STAGE2_ALT_LRS], "stage2_opt")
    best_lr_cfg = select_best([best_s1] + s2_lr_results)["config"]

    _execute(
        [
            {**best_lr_cfg, "optimizer": "rmsprop"},
            {**best_lr_cfg, "optimizer": "sgd", "lr": STAGE2_SGD_LR},
            {**best_lr_cfg, "dropout": STAGE2_DROPOUT},
            {**best_lr_cfg, "bidirectional": True},
        ],
        "stage2_opt",
    )

    best = select_best(all_results)
    out_dir = Path(results_root) / DATASET_NAME / model_type

    rows = []
    for r in all_results:
        cfg = r["config"]
        rows.append(
            {
                "stage": r["stage"],
                "hidden_size": int(cfg["hidden_size"]),
                "num_layers": int(cfg["num_layers"]),
                "bidirectional": bool(cfg["bidirectional"]),
                "dropout": float(cfg["dropout"]),
                "optimizer": cfg["optimizer"],
                "lr": float(cfg["lr"]),
                "batch_size": int(cfg["batch_size"]),
                "n_params": int(r["param_count"]),
                "epochs": int(r["epochs"]),
                "converged": bool(r["converged"]),
                "diverged": bool(r.get("diverged", False)),
                "stop_reason": r["stop_reason"],
                "train_error": round(float(r["train_error"]), 6) if np.isfinite(r["train_error"]) else float("nan"),
                "train_acc": round(float(r["train_acc"]), 6),
                "val_loss": round(float(r["val_loss"]), 6) if np.isfinite(r["val_loss"]) else float("inf"),
                "val_acc": round(float(r["val_acc"]), 6),
                "test_acc": round(float(r["test_acc"]), 6),
                "run_id": r["run_id"],
                "run_dir": r["run_dir"],
                "config": cfg,
            }
        )

    table_cols = [c for c in rows[0].keys() if c not in ("config", "run_id", "run_dir")]
    save_csv_and_md(pd.DataFrame(rows)[table_cols], out_dir / "all_configs")
    save_json(rows, out_dir / "all_configs.json")

    ckpt_path = Path(models_dir) / f"{DATASET_NAME}_{model_type}_best.pt"
    if best["run_id"] in states_by_run:
        save_checkpoint(ckpt_path, states_by_run[best["run_id"]], best["config"], prep, int(best["param_count"]))

    details = {
        "dataset": DATASET_NAME,
        "model": model_type,
        "stage": best["stage"],
        "config": best["config"],
        "n_params": int(best["param_count"]),
        "epochs": int(best["epochs"]),
        "converged": bool(best["converged"]),
        "stop_reason": best["stop_reason"],
        "train_error": best["train_error"],
        "train_acc": best["train_acc"],
        "val_loss": best["val_loss"],
        "val_acc": best["val_acc"],
        "test_loss": best.get("test_loss"),
        "test_acc": best["test_acc"],
        "class_names": best.get("class_names", prep.class_names),
        "train_cm": best.get("train_cm", []),
        "test_cm": best.get("test_cm", []),
        "test_cm_normalized": best.get("test_cm_normalized", []),
        "test_classification_report_text": best.get("test_classification_report_text", ""),
        "run_id": best["run_id"],
        "run_dir": best["run_dir"],
        "checkpoint_path": str(ckpt_path) if ckpt_path.is_file() else best.get("checkpoint_path"),
    }
    save_json(details, out_dir / "best_model_details.json")

    best_cfg = best["config"]
    best_df = pd.DataFrame(
        [
            {
                "dataset": DATASET_NAME,
                "model": model_type,
                "hidden_size": best_cfg["hidden_size"],
                "num_layers": best_cfg["num_layers"],
                "bidirectional": best_cfg["bidirectional"],
                "dropout": best_cfg["dropout"],
                "optimizer": best_cfg["optimizer"],
                "lr": best_cfg["lr"],
                "batch_size": best_cfg["batch_size"],
                "n_params": details["n_params"],
                "epochs": details["epochs"],
                "stop_reason": details["stop_reason"],
                "train_acc": f"{details['train_acc'] * 100:.2f}%",
                "val_acc": f"{details['val_acc'] * 100:.2f}%",
                "test_acc": f"{details['test_acc'] * 100:.2f}%",
            }
        ]
    )
    save_csv_and_md(best_df, out_dir / "best_model_summary")

    report_text = "\n".join(
        [
            f"Dataset: {DATASET_NAME} | Model: {model_type}",
            f"Best Config (selected by val_acc): {best_cfg}",
            f"Train Acc: {details['train_acc'] * 100:.2f}% | Val Acc: {details['val_acc'] * 100:.2f}% | Test Acc: {details['test_acc'] * 100:.2f}%",
            f"Epochs: {details['epochs']} ({details['stop_reason']})",
            "",
            "Per-Class Test Classification Report:",
            details["test_classification_report_text"],
        ]
    )
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    logger.info(
        "Best (%s/%s): %s | train=%.2f%% val=%.2f%% test=%.2f%%",
        DATASET_NAME,
        model_type,
        best["run_id"],
        best["train_acc"] * 100.0,
        best["val_acc"] * 100.0,
        best["test_acc"] * 100.0,
    )
    return best


def run_part_1(
    model: str = "both",
    seed: int = 42,
    max_epochs: int = 3000,
    tol: float = 1e-4,
    split_seed: int = DEFAULT_SPLIT_SEED,
    force: bool = False,
    plots_only: bool = False,
    device: str = "auto",
    data_root: str | Path = DATA_ROOT,
    results_root: str | Path = RESULTS_ROOT,
    models_dir: str | Path = MODELS_DIR,
) -> None:
    data_root_p = Path(data_root)
    results_root_p = Path(results_root)
    models_dir_p = Path(models_dir)

    models_dir_p.mkdir(parents=True, exist_ok=True)
    results_root_p.mkdir(parents=True, exist_ok=True)

    if not plots_only:
        prep = prepare_dataloaders(
            dataset=DATASET_NAME,
            data_root=data_root_p,
            results_root=results_root_p,
            batch_size=32,
            split_seed=split_seed,
            seed=seed,
        )

        model_types: List[str] = ["rnn", "lstm"] if model == "both" else [model]
        for m_type in model_types:
            logger.info("Running %s sweep for %s", m_type.upper(), DATASET_NAME)
            run_sweep(
                model_type=m_type,
                prep=prep,
                seed=seed,
                max_epochs=max_epochs,
                tol=tol,
                split_seed=split_seed,
                data_root=data_root_p,
                results_root=results_root_p,
                models_dir=models_dir_p,
                force=force,
                device_arg=device,
            )

    digest = generate_all_plots_for_dataset(
        data_root=data_root_p,
        results_root=results_root_p,
        dataset=DATASET_NAME,
    )
    update_summary_markdown(results_root_p, digests={DATASET_NAME: digest})
    logger.info("Part 1 finished. Results saved in %s", results_root_p / DATASET_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 6 Part 1: Handwritten Character Pipeline")
    parser.add_argument("--model", type=str, default="both", choices=["both", "rnn", "lstm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plots_only", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--models_dir", type=str, default=str(MODELS_DIR))
    args = parser.parse_args()

    run_part_1(
        model=args.model,
        seed=args.seed,
        max_epochs=args.max_epochs,
        tol=args.tol,
        split_seed=args.split_seed,
        force=args.force,
        plots_only=args.plots_only,
        device=args.device,
        data_root=args.data_root,
        results_root=args.results_root,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
