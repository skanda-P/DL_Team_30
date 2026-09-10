import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Ensure src/ is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DEFAULT_RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

from data_loader import get_data_tensors
from models import ARCHITECTURES, ARCH_GROUPS, ACTIVATION_CHOICES, build_model
from train import OPTIMIZERS, train_single_run, evaluate
from utils.metrics import classification_metrics, confusion_matrix, print_classification_report
from utils.plotting import (
    plot_superimposed_error_vs_epochs,
    plot_confusion_matrix_heatmap,
    plot_convergence_bar_chart
)
from utils.training_config import MAX_EPOCHS, STOPPING_THRESHOLD, PATIENCE


def run_experiments(architectures=None, activations=None, optimizers=None, data_dir=None, results_dir=None,
                    stopping_threshold=STOPPING_THRESHOLD, max_epochs=MAX_EPOCHS,
                    patience=1, device=None, seed=42):
    """
    Executes the full experiment suite across architectures, activations, and optimizers.
    Generates:
      1. Superimposed error vs. epochs plots for each (architecture, activation) pair.
      2. Comprehensive comparison table of convergence epochs, train/val accuracy across all activations.
      3. Identification of best overall architecture & activation, evaluating test accuracy and confusion matrices.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    if architectures is None:
        architectures = ARCH_GROUPS["all"]
    if activations is None:
        activations = list(ACTIVATION_CHOICES)

    if optimizers is None:
        optimizers = list(OPTIMIZERS.keys())
    if device is None:
        device = "cpu"

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("CS601T Deep Learning Assignment 3 - Full Experiment Suite")
    print(f"Architectures: {architectures}")
    print(f"Activations:   {activations}")
    print(f"Optimizers:    {optimizers}")
    print(f"Total Runs:    {len(architectures) * len(activations) * len(optimizers)}")
    print(f"Device:        {device}")
    max_epochs_str = str(max_epochs) if max_epochs is not None else "Unlimited"
    print(f"Max Epochs:    {max_epochs_str} | Patience: {patience}")
    print("=" * 70)

    all_results = []
    epochs_summary = {}
    best_val_acc = -1.0
    best_run = None

    for arch_name in architectures:
        for act_name in activations:
            arch_act_losses = {}
            config_label = f"{arch_name}_{act_name}"
            epochs_summary[config_label] = {}
            print(f"\n>>>>>>>>>>>> Architecture: {arch_name.upper()} | Activation: {act_name.upper()} <<<<<<<<<<<<")

            for opt_key in optimizers:
                res = train_single_run(
                    arch_name=arch_name,
                    optimizer_key=opt_key,
                    activation=act_name,
                    data_dir=data_dir,
                    results_dir=results_dir,
                    stopping_threshold=stopping_threshold,
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                    verbose=True,
                    seed=seed
                )

                all_results.append(res)
                arch_act_losses[res["display_name"]] = res["losses"]
                epochs_summary[config_label][res["display_name"]] = res["epochs_run"]

                if res["val_acc"] > best_val_acc:
                    best_val_acc = res["val_acc"]
                    best_run = res

            # Presentation of Results #2: Superimposed error vs epochs plot for this (arch, act)
            superimposed_plot_path = os.path.join(results_dir, f"{arch_name}_{act_name}_superimposed_error.png")
            plot_superimposed_error_vs_epochs(
                optimizer_losses=arch_act_losses,
                title=f"Average Training Error vs. Epochs: {arch_name.upper()} ({act_name.upper()})",
                filename=superimposed_plot_path
            )
            print(f"[+] Saved superimposed plot to: {superimposed_plot_path}")

    # Plot convergence comparison across configurations and optimizers
    if len(epochs_summary) > 0 and len(optimizers) > 0:
        bar_plot_path = os.path.join(results_dir, "convergence_comparison_bar.png")
        plot_convergence_bar_chart(
            epochs_summary=epochs_summary,
            title="Epochs to Convergence by Architecture, Activation & Optimizer",
            filename=bar_plot_path
        )
        print(f"[+] Saved convergence bar chart to: {bar_plot_path}")

    # Presentation of Results #1 & #3: Tabulate and compare
    summary_table_md, summary_table_csv = generate_comparison_tables(all_results)
    
    with open(os.path.join(results_dir, "summary_table.md"), "w") as f:
        f.write(summary_table_md)
    with open(os.path.join(results_dir, "summary_table.csv"), "w") as f:
        f.write(summary_table_csv)

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY TABLE")
    print("=" * 80)
    print(summary_table_md)

    # Presentation of Results #4: Evaluate best architecture
    evaluate_best_architecture(best_run, data_dir=data_dir, results_dir=results_dir, device=device)

    print("\n[+] Full experimentation suite completed successfully!")

    return all_results, best_run


def generate_comparison_tables(results):
    """
    Generates Markdown and CSV formatted comparison tables.
    """
    headers = [
        "Architecture", "Activation", "Optimizer", "Converged (Epochs)",
        "Stopped by Threshold?", "Train Loss", "Train Acc (%)",
        "Val Loss", "Val Acc (%)", "Val Macro F1 (%)", "Time (s)"
    ]

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    csv_lines = [",".join(headers)]

    for r in results:
        row = [
            r["arch"],
            r.get("activation", "relu"),
            r["display_name"],
            str(r["epochs_run"]),
            "Yes" if r["converged"] else "No (max)",
            f"{r['train_loss']:.4f}",
            f"{r['train_acc'] * 100:.2f}",
            f"{r['val_loss']:.4f}",
            f"{r['val_acc'] * 100:.2f}",
            f"{r['val_metrics']['macro_f_measure'] * 100:.2f}",
            f"{r['elapsed_time']:.2f}"
        ]
        md_lines.append("| " + " | ".join(row) + " |")
        csv_lines.append(",".join(row))

    return "\n".join(md_lines), "\n".join(csv_lines)


def evaluate_best_architecture(best_run, data_dir=None, results_dir=None, device="cpu"):
    """
    Evaluates best architecture and activation on test set and train set.
    Generates confusion matrices and classification reports (Presentation of Results #4).
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    if best_run is None:
        print("No best run identified.")
        return

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION EVALUATION (Presentation of Results Requirement 4)")
    print("=" * 80)
    print(f"Selected Best Configuration:")
    print(f"  Architecture: {best_run['arch']}")
    print(f"  Activation:   {best_run.get('activation', 'N/A')}")
    print(f"  Optimizer:    {best_run['display_name']}")
    print(f"  Val Accuracy: {best_run['val_acc'] * 100:.2f}%")
    print(f"  Epochs to Convergence: {best_run['epochs_run']}")

    best_model = best_run["model"].to(device)
    criterion = torch.nn.CrossEntropyLoss()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx, idx_to_class = get_data_tensors(
        data_dir=data_dir, device=device
    )
    num_classes = len(class_to_idx)
    class_names = [f"Digit {idx_to_class[i]}" for i in range(num_classes)]

    # Evaluate on Test Split
    test_loss, test_acc, test_preds = evaluate(best_model, X_test, y_test, criterion)
    test_cm = confusion_matrix(y_test.cpu().numpy(), test_preds, num_classes)
    test_metrics = classification_metrics(y_test.cpu().numpy(), test_preds, num_classes)

    # Evaluate on Train Split
    train_loss, train_acc, train_preds = evaluate(best_model, X_train, y_train, criterion)
    train_cm = confusion_matrix(y_train.cpu().numpy(), train_preds, num_classes)
    train_metrics = classification_metrics(y_train.cpu().numpy(), train_preds, num_classes)

    print(f"\n--- Train Set Performance ---")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print("Train Confusion Matrix:")
    print(train_cm)

    print(f"\n--- Test Set Performance ---")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print("Test Confusion Matrix:")
    print(test_cm)
    print("\nDetailed Test Classification Report:")
    print_classification_report(test_metrics)

    # Save Confusion Matrix Plots
    best_dir = os.path.join(results_dir, "best_architecture")
    os.makedirs(best_dir, exist_ok=True)

    act_str = best_run.get('activation', 'relu').upper()
    test_cm_plot = os.path.join(best_dir, "test_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        cm=test_cm,
        class_names=class_names,
        title=f"Test Confusion Matrix ({best_run['arch'].upper()} - {act_str} - {best_run['display_name']})",
        filename=test_cm_plot
    )
    print(f"[+] Saved Test Confusion Matrix plot to: {test_cm_plot}")

    train_cm_plot = os.path.join(best_dir, "train_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        cm=train_cm,
        class_names=class_names,
        title=f"Train Confusion Matrix ({best_run['arch'].upper()} - {act_str} - {best_run['display_name']})",
        filename=train_cm_plot
    )
    print(f"[+] Saved Train Confusion Matrix plot to: {train_cm_plot}")

    # Save summary markdown report
    report_path = os.path.join(results_dir, "best_architecture_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Best Architecture Evaluation Report\n\n")
        f.write(f"- **Architecture**: {best_run['arch']}\n")
        f.write(f"- **Activation**: {best_run.get('activation', 'N/A')}\n")
        f.write(f"- **Optimizer**: {best_run['display_name']}\n")
        f.write(f"- **Convergence Epochs**: {best_run['epochs_run']}\n")
        f.write(f"- **Training Accuracy**: {train_acc * 100:.2f}%\n")
        f.write(f"- **Validation Accuracy**: {best_run['val_acc'] * 100:.2f}%\n")
        f.write(f"- **Test Accuracy**: {test_acc * 100:.2f}%\n\n")
        f.write(f"### Test Confusion Matrix\n\n```\n{test_cm}\n```\n\n")
        f.write(f"### Train Confusion Matrix\n\n```\n{train_cm}\n```\n\n")
        f.write(f"### Test Macro Metrics\n\n")
        f.write(f"- Macro Precision: {test_metrics['macro_precision']:.4f}\n")
        f.write(f"- Macro Recall: {test_metrics['macro_recall']:.4f}\n")
        f.write(f"- Macro F1-Score: {test_metrics['macro_f_measure']:.4f}\n")
    print(f"[+] Saved Best Architecture Report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete optimization experimentation suite.")
    parser.add_argument("--arch", type=str, nargs="+", default=None,
                        help="Specific architecture(s) or 'all' (e.g. arch1 arch2)")
    parser.add_argument("--activation", type=str, nargs="+", default=None,
                        help="Specific activation(s) or 'all' (choices: relu tanh sigmoid)")
    parser.add_argument("--optimizer", type=str, nargs="+", default=None,
                        help="Specific optimizer(s) or 'all' (e.g. bgd rmsprop adagrad)")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=STOPPING_THRESHOLD,
                        help="Stopping threshold |L_t - L_{t-1}| < threshold")
    def parse_max_epochs(v):
        if v is None or str(v).lower() in ("none", "inf", "unlimited", "0", "-1"):
            return None
        return int(v)

    parser.add_argument("--max_epochs", type=parse_max_epochs, default=MAX_EPOCHS,
                        help="Maximum epochs per run (default: 10000, or 'none' for unlimited)")
    parser.add_argument("--patience", type=int, default=1, help="Consecutive epochs below threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")

    args = parser.parse_args()

    if args.arch is None or "all" in args.arch:
        archs = ARCH_GROUPS["all"]
    else:
        archs = []
        for a in args.arch:
            if a in ARCH_GROUPS:
                archs.extend(ARCH_GROUPS[a])
            else:
                archs.append(a)

    acts = list(ACTIVATION_CHOICES) if (args.activation is None or "all" in args.activation) else args.activation
    opts = list(OPTIMIZERS.keys()) if (args.optimizer is None or "all" in args.optimizer) else args.optimizer



    run_experiments(
        architectures=archs,
        activations=acts,
        optimizers=opts,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        stopping_threshold=args.threshold,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

