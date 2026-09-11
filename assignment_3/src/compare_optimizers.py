import os
import sys
import json
import time
import argparse
import numpy as np
import torch

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

    all_results = []
    epochs_summary = {}
    best_val_acc = -1.0
    best_run = None

    for arch_name in architectures:
        for act_name in activations:
            arch_act_losses = {}
            config_label = arch_name if len(activations) == 1 else f"{arch_name}_{act_name}"
            epochs_summary[config_label] = {}

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

            plot_filename = f"{arch_name}_superimposed_error.png" if len(activations) == 1 else f"{arch_name}_{act_name}_superimposed_error.png"
            superimposed_plot_path = os.path.join(results_dir, plot_filename)
            plot_title = f"Average Training Error vs. Epochs: {arch_name.upper()}" if len(activations) == 1 else f"Average Training Error vs. Epochs: {arch_name.upper()} ({act_name.upper()})"
            plot_superimposed_error_vs_epochs(
                optimizer_losses=arch_act_losses,
                title=plot_title,
                filename=superimposed_plot_path
            )

    if len(epochs_summary) > 0 and len(optimizers) > 0:
        bar_plot_path = os.path.join(results_dir, "convergence_comparison_bar.png")
        plot_convergence_bar_chart(
            epochs_summary=epochs_summary,
            title="Epochs to Convergence Across Architectures and Optimizers (ReLU)",
            filename=bar_plot_path
        )

    summary_table_md, summary_table_csv = generate_comparison_tables(all_results)
    with open(os.path.join(results_dir, "summary_table.md"), "w") as f:
        f.write(summary_table_md)
    with open(os.path.join(results_dir, "summary_table.csv"), "w") as f:
        f.write(summary_table_csv)

    summary_text = generate_text_summary(all_results, best_run=best_run)
    summary_txt_path = os.path.join(results_dir, "summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_table_md)

    evaluate_best_architecture(best_run, data_dir=data_dir, results_dir=results_dir, device=device)

    return all_results, best_run


def generate_comparison_tables(results):
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
            "Yes" if r.get("converged", False) else "No (max)",
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


def generate_text_summary(results, best_run=None):
    lines = []
    lines.append("=" * 115)
    lines.append("CS601T DEEP LEARNING ASSIGNMENT 3: OPTIMIZATION AND MODEL EVALUATION SUMMARY REPORT")
    lines.append("Team: DL_Team_30 (Group 30)")
    lines.append("Dataset: 5-Class MNIST Sub-dataset (Classes: ['3', '4', '5', '7', '8']) | Input Dim: 784 | Output Dim: 5")
    lines.append("Activation Function: Standardized ReLU across all architectures and optimizers")
    lines.append("=" * 115)
    lines.append("")

    if not results:
        lines.append("No experiment results available.")
        return "\n".join(lines)

    lines.append("=" * 115)
    lines.append("SECTION 1: OVERALL MODEL COMPARISON TABLE (ALL CONFIGURATIONS)")
    lines.append("=" * 115)
    header = f"{'Architecture':<13} {'Activation':<11} {'Optimizer':<25} {'Epochs':<8} {'Stopped?':<10} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<10} {'Val Acc':<11} {'Time (s)':<9}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        arch = r["arch"]
        act = r.get("activation", "relu")
        opt = r["display_name"]
        ep = str(r["epochs_run"])
        stopped = "Yes" if r.get("converged", False) else "No (max)"
        tr_loss = f"{r['train_loss']:.4f}"
        tr_acc = f"{r['train_acc'] * 100:.2f}%"
        val_loss = f"{r['val_loss']:.4f}"
        val_acc = f"{r['val_acc'] * 100:.2f}%"
        el_time = f"{r['elapsed_time']:.2f}"
        lines.append(f"{arch:<13} {act:<11} {opt:<25} {ep:<8} {stopped:<10} {tr_loss:<12} {tr_acc:<11} {val_loss:<10} {val_acc:<11} {el_time:<9}")

    lines.append("")

    distinct_acts = set(r.get("activation", "relu").lower() for r in results)
    sec_num = 2

    if len(distinct_acts) > 1:
        lines.append("=" * 115)
        lines.append(f"SECTION {sec_num}: ACTIVATION FUNCTION COMPARISON (KEEPING ARCHITECTURE & OPTIMIZER CONSTANT)")
        lines.append("=" * 115)
        lines.append("Compares activations under identical model architectures and optimization algorithms.\n")
        sec_num += 1

        act_groups = {}
        for r in results:
            key = (r["arch"], r["display_name"])
            if key not in act_groups:
                act_groups[key] = []
            act_groups[key].append(r)

        act_stats = {}
        for (arch, opt), group in sorted(act_groups.items()):
            lines.append(f"-> Architecture: {arch} | Optimizer: {opt}")
            best_g = max(group, key=lambda x: x["val_acc"])
            for r in sorted(group, key=lambda x: x.get("activation", "")):
                act = r.get("activation", "relu").upper()
                val_acc = r["val_acc"] * 100
                ep = r["epochs_run"]
                t_loss = r["train_loss"]
                v_loss = r["val_loss"]
                tm = r["elapsed_time"]
                lines.append(f"   [{act:<7}] Val Accuracy: {val_acc:6.2f}% | Epochs to Converge: {ep:4d} | Val Loss: {v_loss:.4f} | Train Loss: {t_loss:.4f} | Time: {tm:7.2f}s")

                act_name = r.get("activation", "relu").lower()
                if act_name not in act_stats:
                    act_stats[act_name] = {"val_accs": [], "epochs": [], "losses": []}
                act_stats[act_name]["val_accs"].append(val_acc)
                act_stats[act_name]["epochs"].append(ep)
                act_stats[act_name]["losses"].append(v_loss)

            if len(group) > 1:
                lines.append(f"   >>> Highest Val Accuracy: {best_g.get('activation', '').upper()} ({best_g['val_acc']*100:.2f}%)\n")
            else:
                lines.append("")

        if act_stats:
            lines.append("--- Aggregate Performance by Activation Function ---")
            for act, data in sorted(act_stats.items(), key=lambda x: -np.mean(x[1]["val_accs"])):
                mean_acc = np.mean(data["val_accs"])
                mean_ep = np.mean(data["epochs"])
                mean_loss = np.mean(data["losses"])
                lines.append(f"  * {act.upper():<8}: Mean Val Acc = {mean_acc:6.2f}% | Mean Epochs = {mean_ep:5.1f} | Mean Val Loss = {mean_loss:.4f} (from {len(data['val_accs'])} runs)")
            lines.append("")

    lines.append("=" * 115)
    lines.append(f"SECTION {sec_num}: MODEL ARCHITECTURE COMPARISON (KEEPING OPTIMIZER CONSTANT)")
    lines.append("=" * 115)
    lines.append("Compares the 9 architectures (3 depths x 3 variants) under identical optimizers (using ReLU).\n")
    sec_num += 1

    arch_groups = {}
    for r in results:
        key = (r.get("activation", "relu"), r["display_name"])
        if key not in arch_groups:
            arch_groups[key] = []
        arch_groups[key].append(r)

    arch_stats = {}
    depth_stats = {3: [], 4: [], 5: []}

    for (act, opt), group in sorted(arch_groups.items()):
        act_label = f" | Activation: {act.upper()}" if len(distinct_acts) > 1 else ""
        lines.append(f"-> Optimizer: {opt}{act_label}")
        best_g = max(group, key=lambda x: x["val_acc"])
        for r in sorted(group, key=lambda x: x["arch"]):
            arch = r["arch"]
            val_acc = r["val_acc"] * 100
            ep = r["epochs_run"]
            tm = r["elapsed_time"]
            lines.append(f"   [{arch:<11}] Val Accuracy: {val_acc:6.2f}% | Epochs to Converge: {ep:4d} | Time: {tm:7.2f}s")

            if arch not in arch_stats:
                arch_stats[arch] = {"val_accs": [], "epochs": []}
            arch_stats[arch]["val_accs"].append(val_acc)
            arch_stats[arch]["epochs"].append(ep)

            if "3l" in arch or arch == "arch1":
                depth_stats[3].append(val_acc)
            elif "4l" in arch or arch == "arch2":
                depth_stats[4].append(val_acc)
            elif "5l" in arch or arch == "arch3":
                depth_stats[5].append(val_acc)

        if len(group) > 1:
            lines.append(f"   >>> Highest Val Accuracy: {best_g['arch']} ({best_g['val_acc']*100:.2f}%)\n")
        else:
            lines.append("")

    if arch_stats:
        lines.append("--- Aggregate Performance by Model Architecture ---")
        for arch, data in sorted(arch_stats.items(), key=lambda x: -np.mean(x[1]["val_accs"])):
            mean_acc = np.mean(data["val_accs"])
            mean_ep = np.mean(data["epochs"])
            lines.append(f"  * {arch:<12}: Mean Val Acc = {mean_acc:6.2f}% | Mean Epochs = {mean_ep:5.1f} (from {len(data['val_accs'])} runs)")

        lines.append("\n--- Aggregate Performance by Hidden Layer Depth ---")
        for d in [3, 4, 5]:
            if depth_stats[d]:
                lines.append(f"  * {d} Hidden Layers: Mean Val Acc = {np.mean(depth_stats[d]):6.2f}% (from {len(depth_stats[d])} runs)")
        lines.append("\nObservations:")
        lines.append("  - 3-layer networks provide high layer capacities (e.g. 512-256-128) and short backpropagation paths, resulting in strong baseline performance.")
        lines.append("  - 4-layer and 5-layer variants provide deeper hierarchical representations, but require adaptive optimizers like Adam/RMSProp to converge robustly.\n")

    lines.append("=" * 115)
    lines.append(f"SECTION {sec_num}: GRADIENT DESCENT METHOD COMPARISON (KEEPING ARCHITECTURE CONSTANT)")
    lines.append("=" * 115)
    lines.append("Compares all 7 optimization algorithms under identical architectures (using ReLU).\n")
    sec_num += 1

    opt_groups = {}
    for r in results:
        key = (r["arch"], r.get("activation", "relu"))
        if key not in opt_groups:
            opt_groups[key] = []
        opt_groups[key].append(r)

    opt_stats = {}

    for (arch, act), group in sorted(opt_groups.items()):
        act_label = f" | Activation: {act.upper()}" if len(distinct_acts) > 1 else ""
        lines.append(f"-> Architecture: {arch}{act_label}")
        best_g = max(group, key=lambda x: x["val_acc"])
        fastest_g = min(group, key=lambda x: x["epochs_run"])
        for r in sorted(group, key=lambda x: x["display_name"]):
            opt = r["display_name"]
            val_acc = r["val_acc"] * 100
            ep = r["epochs_run"]
            tm = r["elapsed_time"]
            v_loss = r["val_loss"]
            lines.append(f"   [{opt:<30}] Val Acc: {val_acc:6.2f}% | Epochs: {ep:4d} | Val Loss: {v_loss:.4f} | Time: {tm:7.2f}s")

            opt_key = r["optimizer"]
            if opt_key not in opt_stats:
                opt_stats[opt_key] = {"display_name": opt, "val_accs": [], "epochs": [], "times": []}
            opt_stats[opt_key]["val_accs"].append(val_acc)
            opt_stats[opt_key]["epochs"].append(ep)
            opt_stats[opt_key]["times"].append(tm)

        if len(group) > 1:
            lines.append(f"   >>> Highest Accuracy:   {best_g['display_name']} ({best_g['val_acc']*100:.2f}%)")
            lines.append(f"   >>> Fastest Convergence: {fastest_g['display_name']} ({fastest_g['epochs_run']} epochs)\n")
        else:
            lines.append("")

    if opt_stats:
        lines.append("--- Aggregate Performance by Optimizer ---")
        for opt_key, data in sorted(opt_stats.items(), key=lambda x: -np.mean(x[1]["val_accs"])):
            mean_acc = np.mean(data["val_accs"])
            mean_ep = np.mean(data["epochs"])
            mean_time = np.mean(data["times"])
            lines.append(f"  * {data['display_name']:<30}: Mean Val Acc = {mean_acc:6.2f}% | Mean Epochs = {mean_ep:5.1f} | Mean Time = {mean_time:7.2f}s")
        lines.append("\nObservations:")
        lines.append("  - Adam delivers superior convergence stability by adjusting per-parameter learning rates using exponential moving averages of gradients and squared gradients.")
        lines.append("  - RMSProp utilizes exponentially decaying squared gradient scaling (beta=0.99) to prevent learning rate vanishing in deep ravines.")
        lines.append("  - Momentum and NAG accelerate vanilla SGD by incorporating velocity vectors, greatly reducing oscillations and speeding up convergence.")
        lines.append("  - Batch GD operates with stable gradient estimates over all N samples per step, but converges more slowly in wall-clock time due to fewer parameter updates per epoch.\n")

    if best_run is not None:
        lines.append("=" * 115)
        lines.append(f"SECTION {sec_num}: OVERALL BEST PERFORMING MODEL IDENTIFICATION (VALIDATION ACCURACY)")
        lines.append("=" * 115)
        lines.append(f"Winning Architecture:        {best_run['arch']}")
        lines.append(f"Winning Activation Function: {best_run.get('activation', 'N/A').upper()}")
        lines.append(f"Winning Optimizer:           {best_run['display_name']}")
        lines.append(f"Validation Accuracy:         {best_run['val_acc'] * 100:.2f}%")
        lines.append(f"Training Accuracy:           {best_run.get('train_acc', 0.0) * 100:.2f}%")
        lines.append(f"Convergence Epochs:          {best_run['epochs_run']}")
        lines.append(f"Convergence Time:            {best_run.get('elapsed_time', 0.0):.2f} seconds")
        lines.append("=" * 115)

    return "\n".join(lines)


def parse_metrics_txt(file_path):
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip()
            if key == "Architecture":
                data["arch"] = val
            elif key in ("Activation Function", "Activation"):
                data["activation"] = val.lower()
            elif key == "Optimizer":
                data["display_name"] = val
                val_low = val.lower()
                if "batch gd" in val_low:
                    data["optimizer"] = "bgd"
                elif "momentum" in val_low:
                    data["optimizer"] = "momentum"
                elif "nag" in val_low:
                    data["optimizer"] = "nag"
                elif "adagrad" in val_low:
                    data["optimizer"] = "adagrad"
                elif "rmsprop" in val_low:
                    data["optimizer"] = "rmsprop"
                elif "adam" in val_low:
                    data["optimizer"] = "adam"
                elif "sgd" in val_low:
                    data["optimizer"] = "sgd"
                else:
                    data["optimizer"] = val_low
            elif key == "Epochs to Converge":
                try:
                    data["epochs_run"] = int(val)
                except ValueError:
                    data["epochs_run"] = 0
            elif key == "Stopped by Threshold":
                data["converged"] = ("yes" in val.lower())
            elif key == "Elapsed Time":
                t_str = val.replace("seconds", "").replace("s", "").strip()
                try:
                    data["elapsed_time"] = float(t_str)
                except ValueError:
                    data["elapsed_time"] = 0.0
            elif key == "Training Loss":
                try:
                    data["train_loss"] = float(val)
                except ValueError:
                    data["train_loss"] = 0.0
            elif key == "Training Accuracy":
                try:
                    data["train_acc"] = float(val.replace("%", "").strip()) / 100.0
                except ValueError:
                    data["train_acc"] = 0.0
            elif key == "Validation Loss":
                try:
                    data["val_loss"] = float(val)
                except ValueError:
                    data["val_loss"] = 0.0
            elif key == "Validation Accuracy":
                try:
                    data["val_acc"] = float(val.replace("%", "").strip()) / 100.0
                except ValueError:
                    data["val_acc"] = 0.0
            elif key == "Validation Macro F1":
                try:
                    macro_f1 = float(val.replace("%", "").strip()) / 100.0
                except ValueError:
                    macro_f1 = 0.0
                if "val_metrics" not in data:
                    data["val_metrics"] = {}
                data["val_metrics"]["macro_f_measure"] = macro_f1
            elif key == "Validation Micro F1":
                try:
                    micro_f1 = float(val.replace("%", "").strip()) / 100.0
                except ValueError:
                    micro_f1 = 0.0
                if "val_metrics" not in data:
                    data["val_metrics"] = {}
                data["val_metrics"]["micro_f_measure"] = micro_f1

    if "val_metrics" not in data:
        data["val_metrics"] = {"macro_f_measure": data.get("val_acc", 0.0), "micro_f_measure": data.get("val_acc", 0.0)}
    return data


def load_results_from_disk(results_dir):
    results = []
    if not os.path.exists(results_dir):
        return results

    for root, dirs, files in os.walk(results_dir):
        if "metrics.txt" in files:
            m_path = os.path.join(root, "metrics.txt")
            parsed = parse_metrics_txt(m_path)
            if parsed and "arch" in parsed:
                results.append(parsed)
        elif "metrics.json" in files:
            m_path = os.path.join(root, "metrics.json")
            try:
                with open(m_path, "r") as f:
                    parsed = json.load(f)
                    if "train_accuracy" in parsed:
                        parsed["train_acc"] = parsed["train_accuracy"]
                    if "val_accuracy" in parsed:
                        parsed["val_acc"] = parsed["val_accuracy"]
                    if "elapsed_time_sec" in parsed:
                        parsed["elapsed_time"] = parsed["elapsed_time_sec"]
                    if "converged_by_threshold" in parsed:
                        parsed["converged"] = parsed["converged_by_threshold"]
                    if "macro_f1" in parsed and "val_metrics" not in parsed:
                        parsed["val_metrics"] = {"macro_f_measure": parsed["macro_f1"], "micro_f_measure": parsed.get("micro_f1", parsed["macro_f1"])}
                    results.append(parsed)
            except Exception:
                pass

    return results


def evaluate_best_architecture(best_run, data_dir=None, results_dir=None, device="cpu"):
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    if best_run is None:
        return

    best_model = best_run["model"].to(device)
    criterion = torch.nn.CrossEntropyLoss()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx, idx_to_class = get_data_tensors(
        data_dir=data_dir, device=device
    )
    num_classes = len(class_to_idx)
    class_names = [f"Digit {idx_to_class[i]}" for i in range(num_classes)]

    test_loss, test_acc, test_preds = evaluate(best_model, X_test, y_test, criterion)
    test_cm = confusion_matrix(y_test.cpu().numpy(), test_preds, num_classes)
    test_metrics = classification_metrics(y_test.cpu().numpy(), test_preds, num_classes)

    train_loss, train_acc, train_preds = evaluate(best_model, X_train, y_train, criterion)
    train_cm = confusion_matrix(y_train.cpu().numpy(), train_preds, num_classes)
    train_metrics = classification_metrics(y_train.cpu().numpy(), train_preds, num_classes)

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

    train_cm_plot = os.path.join(best_dir, "train_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        cm=train_cm,
        class_names=class_names,
        title=f"Train Confusion Matrix ({best_run['arch'].upper()} - {act_str} - {best_run['display_name']})",
        filename=train_cm_plot
    )

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


def main():
    parser = argparse.ArgumentParser(description="Run complete optimization experimentation suite (using ReLU activation).")
    parser.add_argument("--arch", type=str, nargs="+", default=None,
                        help="Specific architecture(s) or 'all' (e.g. arch1 arch2)")
    parser.add_argument("--activation", type=str, nargs="+", default=["relu"],
                        help="Specific activation(s) (default: relu)")
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
    parser.add_argument("--summary_only", action="store_true",
                        help="Only generate/update summary.txt from existing results on disk without training")

    args = parser.parse_args()

    if args.summary_only:
        disk_results = load_results_from_disk(args.results_dir)
        if not disk_results:
            print(f"No experiment results found in '{args.results_dir}' to summarize.")
            return
        best_disk_run = max(disk_results, key=lambda x: x.get("val_acc", 0.0))
        summary_txt = generate_text_summary(disk_results, best_run=best_disk_run)
        summary_path = os.path.join(args.results_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_txt)
        print(summary_txt)
        return

    if args.arch is None or "all" in args.arch:
        archs = ARCH_GROUPS["all"]
    else:
        archs = []
        for a in args.arch:
            if a in ARCH_GROUPS:
                archs.extend(ARCH_GROUPS[a])
            else:
                archs.append(a)

    acts = ["relu"] if (args.activation is None or "all" in args.activation) else args.activation
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
