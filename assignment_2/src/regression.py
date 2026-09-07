# regression.py
# Entry point for the regression pipeline (univariate and bivariate datasets) — FCNN version.
# Same filename/role as Assignment 1, but drives an FCNN sweep instead of a single perceptron.
#
# report_rmse(y_true, y_pred)
#   - Computes RMSE and %RMSE for a given prediction set. Carried over unchanged from Assignment 1.
#
# run_dataset(dataset_name, path, dim, ...)
#   - Splits the data 60/20/20, sweeps 1-hidden-layer (and additionally 2-hidden-layer for the
#     bivariate dataset) architectures, selects the best by validation RMSE, and generates
#     error-vs-epoch, target-vs-model, scatter, and per-node output plots across train/val/test
#     splits for the winning architecture.
#
# main()
#   - Loads the univariate and bivariate datasets and calls run_dataset for each.

import os
import shutil
import numpy as np

import model_selection
import compare_results
from utils.data_utils import load_regression_csv, three_way_split
from utils.metrics import rmse, percent_rmse
from utils.plotting import (
    plot_error_vs_epochs,
    plot_node_output_surface,
    plot_node_output_1d,
    plot_regression_1d,
    plot_regression_2d,
    plot_target_vs_model_scatter,
)


DATA_DIR = "data"
UNIVARIATE_FILE = os.path.join(DATA_DIR, "Regression", "UnivariateData", "30.csv")
BIVARIATE_FILE = os.path.join(DATA_DIR, "Regression", "BivariateData", "30.csv")

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
SEED = 42

# Dataset1 (univariate): FCNN with a single hidden layer only, per the assignment spec.
# Dataset2 (bivariate): both one- and two-hidden-layer architectures are tried.
HIDDEN_LAYER_OPTIONS = {
    "Univariate": [(2,), (4,), (8,), (16,), (32,)],
    "Bivariate": [(4,), (8,), (16,), (32,), (8, 4), (16, 8), (32, 16)],
}
ACTIVATIONS = ["logistic", "tanh"]
LEARNING_RATES = [0.01, 0.05, 0.1]
MAX_EPOCHS = 2000
STOPPING_THRESHOLD = 0.0001
PATIENCE = 5
SELECTION_METRIC = "rmse"


def report_rmse(y_true, y_pred):
    # Computes RMSE and %RMSE for a given prediction set.
    return {
        "rmse": rmse(y_true, y_pred),
        "percent_rmse": percent_rmse(y_true, y_pred),
    }


def save_metrics_file(filepath, dataset_name, cfg, metrics, epochs_run):
    # Writes evaluation metrics in key-value text format
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(
            f"Configuration: {dataset_name} | {cfg['hidden_activation']} | "
            f"LR: {cfg['learning_rate']} | EP: {cfg['epochs']} | "
            f"epochs_run: {epochs_run} | config_id: {cfg['config_id']}\n"
        )
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if isinstance(value, (float, np.floating)):
                f.write(f"{key}: {value:.4f}\n")
            elif isinstance(value, np.ndarray) and value.dtype.kind == "f":
                formatted_arr = "[" + " ".join(f"{v:.4f}" for v in value) + "]"
                f.write(f"{key}: {formatted_arr}\n")
            else:
                f.write(f"{key}: {value}\n")


def save_sweep_metrics_file(filepath, dataset_name, cfg, train_metrics, val_metrics, epochs_run):
    # Sweep entries need BOTH train and validation RMSE/%RMSE (per architecture),
    # unlike the best-model files which store one split's metrics each.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(
            f"Configuration: {dataset_name} | {cfg['hidden_activation']} | "
            f"LR: {cfg['learning_rate']} | EP: {cfg['epochs']} | "
            f"epochs_run: {epochs_run} | config_id: {cfg['config_id']}\n"
        )
        f.write("-" * 40 + "\n")
        f.write("[train]\n")
        for key, value in train_metrics.items():
            if isinstance(value, (float, np.floating)):
                f.write(f"train_{key}: {value:.4f}\n")
            else:
                f.write(f"train_{key}: {value}\n")
        f.write("[validation]\n")
        for key, value in val_metrics.items():
            if isinstance(value, (float, np.floating)):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")


def _plot_fit(X, y_true, y_pred, dim, title, filename):
    # Dispatches to the 1D or 2D target-vs-model fit plot depending on input dimensionality.
    if dim == 1:
        plot_regression_1d(X.reshape(-1), y_true, y_pred, title=title, filename=filename)
    else:
        plot_regression_2d(X, y_true, y_pred, title=title, filename=filename)


def _plot_node_output(X, z, dim, node_label, split_name, filename):
    # Dispatches to the 1D or 2D node-output plot depending on input dimensionality.
    if dim == 1:
        plot_node_output_1d(X, z, node_label, split_name, filename)
    else:
        plot_node_output_surface(X, z, node_label, split_name, filename)


def run_dataset(dataset_name, X, y, dim, hidden_layer_size_options):
    print(f"\n==========================================")
    print(f"Running Regression: {dataset_name}")
    print(f"==========================================")

    # 60/20/20 train/val/test split (pure random, no stratification for continuous targets)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
        X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
    )

    # Generate full architecture sweep grid across activations
    architectures = []
    for act in ACTIVATIONS:
        architectures.extend(
            model_selection.architecture_grid(
                input_dim=X.shape[1],
                output_dim=1,
                hidden_layer_size_options=hidden_layer_size_options,
                hidden_activation=act,
                output_activation="linear",
                learning_rates=LEARNING_RATES,
                max_epochs=MAX_EPOCHS,
                stopping_threshold=STOPPING_THRESHOLD,
                patience=PATIENCE,
            )
        )

    print(f"Total configurations to evaluate: {len(architectures)}")

    # Execute training sweep over validation split
    sweep_results = model_selection.run_regression_sweep(
        X_train, y_train, X_val, y_val, architectures
    )

    # Save metrics for every model in the sweep
    results_base_dir = os.path.abspath(os.path.join("results", dataset_name))
    for res in sweep_results:
        cfg = res["config"]
        cfg_dir = os.path.join(results_base_dir, cfg["config_id"])
        metrics_file = os.path.join(cfg_dir, "evaluation_metrics.txt")
        save_sweep_metrics_file(
            metrics_file, dataset_name, cfg, res["train_metrics"], res["val_metrics"], res["epochs_run"]
        )

    # Pick the best architecture based on validation RMSE (lower is better)
    best = model_selection.select_best_regression(sweep_results, SELECTION_METRIC)
    best_cfg = best["config"]
    best_model = best["model"]
    best_val_metrics = best["val_metrics"]

    print(f"\nBest architecture: {best_cfg['config_id']}")
    print(f"Validation RMSE: {best_val_metrics['rmse']:.4f} "
          f"(%RMSE: {best_val_metrics['percent_rmse']:.2f}%)")
    print(f"Epochs trained: {best['epochs_run']} / {best_cfg['epochs']}")

    # Evaluate best model on train/test splits without retraining (val already computed above)
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    best_train_metrics = report_rmse(y_train, y_train_pred)
    best_test_metrics = report_rmse(y_test, y_test_pred)
    print(f"Test RMSE: {best_test_metrics['rmse']:.4f} "
          f"(%RMSE: {best_test_metrics['percent_rmse']:.2f}%)")

    best_dir = os.path.abspath(os.path.join("results", dataset_name, "best"))
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir, exist_ok=True)

    # Save train, validation, and test metric files for best model
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_train.txt"),
        dataset_name, best_cfg, best_train_metrics, best["epochs_run"],
    )
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_val.txt"),
        dataset_name, best_cfg, best_val_metrics, best["epochs_run"],
    )
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_test.txt"),
        dataset_name, best_cfg, best_test_metrics, best["epochs_run"],
    )

    # 1) Plot error vs epochs for the best model
    plot_error_vs_epochs(
        best_model.errors,
        title=f"{dataset_name} ({best_cfg['config_id']}): Average Error vs Epochs",
        filename=os.path.join(best_dir, "error_vs_epochs.png"),
    )

    splits = [
        ("train", X_train, y_train, y_train_pred),
        ("val", X_val, y_val, y_val_pred),
        ("test", X_test, y_test, y_test_pred),
    ]

    # 3) Model output superimposed on target output, and
    # 4) Scatter plot of target output vs model output — for train/val/test
    for split_name, X_split, y_split, y_pred_split in splits:
        _plot_fit(
            X_split, y_split, y_pred_split, dim,
            title=f"{dataset_name} ({best_cfg['config_id']}): Target vs Model Output ({split_name})",
            filename=os.path.join(best_dir, f"fit_{split_name}.png"),
        )
        plot_target_vs_model_scatter(
            y_split, y_pred_split,
            title=f"{dataset_name} ({best_cfg['config_id']}): Target vs Model Scatter ({split_name})",
            filename=os.path.join(best_dir, f"scatter_{split_name}.png"),
        )

    # 5) Plots of outputs for each hidden node and the output node, for train/val/test
    for split_name, X_split, _, _ in splits:
        activations = best_model.forward_all(X_split)

        # Hidden layers are indices 1 through len(layer_sizes) - 2
        for layer_idx in range(1, len(best_model.layer_sizes) - 1):
            layer_acts = activations[layer_idx]
            for node_idx in range(layer_acts.shape[1]):
                fn = os.path.join(
                    best_dir, f"node_output_{split_name}_hidden{layer_idx}-{node_idx}.png"
                )
                label = f"Hidden Layer {layer_idx} Node {node_idx}"
                _plot_node_output(X_split, layer_acts[:, node_idx], dim, label, split_name, fn)

        # Output layer (single linear node for regression)
        output_acts = activations[-1]
        for node_idx in range(output_acts.shape[1]):
            fn = os.path.join(
                best_dir, f"node_output_{split_name}_output{node_idx}.png"
            )
            label = f"Output Node {node_idx}"
            _plot_node_output(X_split, output_acts[:, node_idx], dim, label, split_name, fn)

    return {
        "dataset_name": dataset_name,
        "best_config": best_cfg,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "epochs_run": best["epochs_run"],
    }


def main():
    X_uni, y_uni = load_regression_csv(UNIVARIATE_FILE)
    X_bi, y_bi = load_regression_csv(BIVARIATE_FILE)

    uni_summary = run_dataset(
        "Univariate", X_uni, y_uni, dim=1,
        hidden_layer_size_options=HIDDEN_LAYER_OPTIONS["Univariate"],
    )
    bi_summary = run_dataset(
        "Bivariate", X_bi, y_bi, dim=2,
        hidden_layer_size_options=HIDDEN_LAYER_OPTIONS["Bivariate"],
    )

    # Generate overall comparison summary report following Assignment 1 pattern
    compare_results.generate_comparison_report()


if __name__ == "__main__":
    main()