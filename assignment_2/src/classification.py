import os
import numpy as np

import model_selection
from utils.data_utils import load_LS_data, load_nls_data, stratified_three_way_split
from utils.metrics import classification_metrics, print_classification_report
from utils.plotting import (
    plot_decision_regions,
    plot_error_vs_epochs,
    plot_node_output_surface,
)


DATA_DIR = "data"
LS_DIR = os.path.join(DATA_DIR, "Classification", "LS_Group30")
NLS_FILE = os.path.join(DATA_DIR, "Classification", "NLS_Group30.txt")

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
SEED = 42

HIDDEN_LAYER_OPTIONS = {
    "LS": [(4,), (8,), (16,), (32,)],
    "NLS": [(8, 4), (16, 8), (32, 16)],
}
ACTIVATIONS = ["logistic", "tanh"]
LEARNING_RATES = [0.05, 0.1]
EPOCHS_LIST = [300, 600]
STOPPING_THRESHOLD = 1e-4
SELECTION_METRIC = "overall_accuracy"


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
            f.write(f"{key}: {value}\n")


def run_dataset(dataset_name, X, y, hidden_layer_size_options):
    print(f"\n==========================================")
    print(f"Running Classification: {dataset_name}")
    print(f"==========================================")

    # 60/20/20 train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(
        X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
    )

    all_classes = np.unique(y)
    num_classes = len(all_classes)

    # Generate full architecture sweep grid across activations
    architectures = []
    for act in ACTIVATIONS:
        architectures.extend(
            model_selection.architecture_grid(
                input_dim=X.shape[1],
                output_dim=num_classes,
                hidden_layer_size_options=hidden_layer_size_options,
                hidden_activation=act,
                output_activation=act,
                learning_rates=LEARNING_RATES,
                epochs_list=EPOCHS_LIST,
                stopping_threshold=STOPPING_THRESHOLD,
            )
        )

    print(f"Total configurations to evaluate: {len(architectures)}")

    # Execute training sweep over validation split
    sweep_results = model_selection.run_sweep(
        X_train, y_train, X_val, y_val, architectures, num_classes
    )

    # Save metrics and decision regions for every model in the sweep
    sweep_base_dir = os.path.abspath(os.path.join("results", dataset_name, "sweep"))
    for res in sweep_results:
        cfg = res["config"]
        cfg_dir = os.path.join(sweep_base_dir, cfg["config_id"])
        metrics_file = os.path.join(cfg_dir, "evaluation_metrics.txt")
        save_metrics_file(
            metrics_file, dataset_name, cfg, res["val_metrics"], res["epochs_run"]
        )

        decision_plot_file = os.path.join(cfg_dir, "decision_region.png")
        plot_decision_regions(
            X_train,
            y_train,
            res["model"].predict,
            title=f"{dataset_name} ({cfg['config_id']}): Decision Region",
            filename=decision_plot_file,
            all_classes=all_classes,
        )

    # Pick the best architecture based on validation metrics
    best = model_selection.select_best(sweep_results, SELECTION_METRIC)
    best_cfg = best["config"]
    best_model = best["model"]
    best_val_metrics = best["val_metrics"]

    print(f"\nBest architecture: {best_cfg['config_id']}")
    print(f"Validation {SELECTION_METRIC}: {best_val_metrics[SELECTION_METRIC]:.4f}")
    print(f"Epochs trained: {best['epochs_run']} / {best_cfg['epochs']}")

    # Evaluate best model on test split without retraining
    y_test_pred = best_model.predict(X_test)
    best_test_metrics = classification_metrics(y_test, y_test_pred, num_classes)
    print(f"Test {SELECTION_METRIC}: {best_test_metrics[SELECTION_METRIC]:.4f}")

    best_dir = os.path.abspath(os.path.join("results", dataset_name, "best"))
    os.makedirs(best_dir, exist_ok=True)

    # Save validation and test metric files for best model
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_val.txt"),
        dataset_name,
        best_cfg,
        best_val_metrics,
        best["epochs_run"],
    )
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_test.txt"),
        dataset_name,
        best_cfg,
        best_test_metrics,
        best["epochs_run"],
    )

    # Plot error vs epochs for the best model
    plot_error_vs_epochs(
        best_model.errors,
        title=f"{dataset_name} ({best_cfg['config_id']}): Average Error vs Epochs",
        filename=os.path.join(best_dir, "error_vs_epochs.png"),
    )

    # Plot decision regions for the best model
    plot_decision_regions(
        X_train,
        y_train,
        best_model.predict,
        title=f"{dataset_name} ({best_cfg['config_id']}): Decision Region",
        filename=os.path.join(best_dir, "decision_region.png"),
        all_classes=all_classes,
    )

    # Generate 3D output surfaces for each hidden node and output node
    splits = [("train", X_train), ("val", X_val), ("test", X_test)]
    for split_name, X_split in splits:
        activations = best_model.forward_all(X_split)

        # Hidden layers are indices 1 through len(layer_sizes) - 2
        for layer_idx in range(1, len(best_model.layer_sizes) - 1):
            layer_acts = activations[layer_idx]
            for node_idx in range(layer_acts.shape[1]):
                fn = os.path.join(
                    best_dir, f"node_output_{split_name}_hidden{layer_idx}-{node_idx}.png"
                )
                label = f"Hidden Layer {layer_idx} Node {node_idx}"
                plot_node_output_surface(
                    X_split, layer_acts[:, node_idx], label, split_name, fn
                )

        # Output layer is activations[-1]
        output_acts = activations[-1]
        for node_idx in range(output_acts.shape[1]):
            fn = os.path.join(
                best_dir, f"node_output_{split_name}_output{node_idx}.png"
            )
            label = f"Output Node {node_idx}"
            plot_node_output_surface(
                X_split, output_acts[:, node_idx], label, split_name, fn
            )

    return {
        "dataset_name": dataset_name,
        "best_config": best_cfg,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "epochs_run": best["epochs_run"],
    }


def main():
    X_ls, y_ls = load_LS_data(LS_DIR)
    X_nls, y_nls = load_nls_data(NLS_FILE)

    ls_summary = run_dataset("LS", X_ls, y_ls, HIDDEN_LAYER_OPTIONS["LS"])
    nls_summary = run_dataset("NLS", X_nls, y_nls, HIDDEN_LAYER_OPTIONS["NLS"])

    # Write summary file
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write("=== ASSIGNMENT 2 CLASSIFICATION SUMMARY ===\n\n")
        for s in [ls_summary, nls_summary]:
            f.write(f"Dataset: {s['dataset_name']}\n")
            f.write(f"Best Config ID: {s['best_config']['config_id']}\n")
            f.write(f"Layer Sizes: {s['best_config']['layer_sizes']}\n")
            f.write(f"Activation: {s['best_config']['hidden_activation']}\n")
            f.write(f"Epochs Run: {s['epochs_run']} / {s['best_config']['epochs']}\n")
            f.write(f"Validation Accuracy: {s['best_val_metrics']['overall_accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {s['best_test_metrics']['overall_accuracy']:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"\nSaved overall summary to {summary_path}")


if __name__ == "__main__":
    main()
