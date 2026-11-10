"""2-stage hyperparameter sweep for Assignment 6 RNN and LSTM classifiers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data_loader import DATA_ROOT, DEFAULT_SPLIT_SEED, RESULTS_ROOT, PreparedData, prepare_dataloaders
from train import MODELS_DIR, build_default_config, run_training, save_checkpoint
from utils.io import save_csv_and_md, save_json, setup_logger

logger = setup_logger("run_experiments")

# Stage 1 (9 configs): width x depth
STAGE1_HIDDEN_SIZES = [32, 64, 128]
STAGE1_NUM_LAYERS = [1, 2, 3]

# Stage 2 (6 configs): optimization & regularization tuning around Stage 1 winner
STAGE2_ALT_LRS = [3e-3, 3e-4]
STAGE2_SGD_LR = 1e-2
STAGE2_DROPOUT = 0.2


def select_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select best model strictly using validation accuracy, val loss, and parameter count."""
    return max(
        results,
        key=lambda r: (
            0 if r.get("diverged") else 1,
            float(r["val_acc"]),
            -float(r["val_loss"]) if np.isfinite(r["val_loss"]) else -1e12,
            -int(r["param_count"]),
        ),
    )


def _summary_row(r: Dict[str, Any]) -> Dict[str, Any]:
    cfg = r["config"]
    return {
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


def _save_best_outputs(
    dataset: str,
    model_type: str,
    best: Dict[str, Any],
    best_state: Any,
    prep: PreparedData,
    results_root: str | Path,
    models_dir: str | Path,
) -> None:
    out_dir = Path(results_root) / dataset / model_type
    ckpt_path = Path(models_dir) / f"{dataset}_{model_type}_best.pt"
    if best_state is not None:
        save_checkpoint(ckpt_path, best_state, best["config"], prep, int(best["param_count"]))

    details = {
        "dataset": dataset,
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

    cfg = best["config"]
    best_df = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model_type,
                "hidden_size": cfg["hidden_size"],
                "num_layers": cfg["num_layers"],
                "bidirectional": cfg["bidirectional"],
                "dropout": cfg["dropout"],
                "optimizer": cfg["optimizer"],
                "lr": cfg["lr"],
                "batch_size": cfg["batch_size"],
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

    report = "\n".join(
        [
            f"Dataset: {dataset} | Model: {model_type}",
            f"Best Config (selected by val_acc): {cfg}",
            f"Train Acc: {details['train_acc'] * 100:.2f}% | Val Acc: {details['val_acc'] * 100:.2f}% | Test Acc: {details['test_acc'] * 100:.2f}%",
            f"Epochs: {details['epochs']} ({details['stop_reason']})",
            "",
            "Per-Class Test Classification Report:",
            details["test_classification_report_text"],
        ]
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")


def run_study(
    dataset: str,
    model_type: str,
    seed: int,
    max_epochs: int,
    tol: float,
    split_seed: int,
    data_root: str | Path,
    results_root: str | Path,
    models_dir: str | Path,
    force: bool,
    device_arg: str,
    prep: PreparedData,
) -> Dict[str, Any]:
    """Run all 15 configs (9 Stage 1 + 6 Stage 2) once each and save best model outputs."""
    all_results: List[Dict[str, Any]] = []
    states_by_run: Dict[str, Any] = {}

    def run_configs(cfgs: List[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
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
                dataset,
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

    logger.info("=== Stage 1: Architecture Grid (%s / %s) ===", dataset, model_type)
    s1_cfgs = [
        build_default_config(
            dataset=dataset,
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
    s1_results = run_configs(s1_cfgs, "stage1_arch")
    best_s1 = select_best(s1_results)

    logger.info("=== Stage 2: Tuning LR, Optimizer, Dropout, Bidir (%s / %s) ===", dataset, model_type)
    s2_lr_results = run_configs([{**best_s1["config"], "lr": lr} for lr in STAGE2_ALT_LRS], "stage2_opt")
    best_lr_cfg = select_best([best_s1] + s2_lr_results)["config"]

    run_configs(
        [
            {**best_lr_cfg, "optimizer": "rmsprop"},
            {**best_lr_cfg, "optimizer": "sgd", "lr": STAGE2_SGD_LR},
            {**best_lr_cfg, "dropout": STAGE2_DROPOUT},
            {**best_lr_cfg, "bidirectional": True},
        ],
        "stage2_opt",
    )

    best = select_best(all_results)
    out_dir = Path(results_root) / dataset / model_type
    rows = [_summary_row(r) for r in all_results]
    table_cols = [c for c in rows[0].keys() if c not in ("config", "run_id", "run_dir")]
    save_csv_and_md(pd.DataFrame(rows)[table_cols], out_dir / "all_configs")
    save_json(rows, out_dir / "all_configs.json")
    _save_best_outputs(dataset, model_type, best, states_by_run.get(best["run_id"]), prep, results_root, models_dir)

    logger.info(
        "Best (%s/%s): %s | train=%.2f%% val=%.2f%% test=%.2f%%",
        dataset,
        model_type,
        best["run_id"],
        best["train_acc"] * 100.0,
        best["val_acc"] * 100.0,
        best["test_acc"] * 100.0,
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2-stage hyperparameter sweep (15 configs per model).")
    parser.add_argument("--dataset", type=str, required=True, choices=["handwritten", "cv"])
    parser.add_argument("--model", type=str, default="both", choices=["both", "rnn", "lstm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--models_dir", type=str, default=str(MODELS_DIR))
    args = parser.parse_args()

    prep = prepare_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        results_root=args.results_root,
        batch_size=32,
        split_seed=args.split_seed,
        seed=args.seed,
    )
    models = ["rnn", "lstm"] if args.model == "both" else [args.model]
    for m_type in models:
        run_study(
            dataset=args.dataset,
            model_type=m_type,
            seed=args.seed,
            max_epochs=args.max_epochs,
            tol=args.tol,
            split_seed=args.split_seed,
            data_root=args.data_root,
            results_root=args.results_root,
            models_dir=args.models_dir,
            force=args.force,
            device_arg=args.device,
            prep=prep,
        )


if __name__ == "__main__":
    main()
