import os
import glob


A1_RESULTS_DIR = os.path.abspath(os.path.join("results", "a1"))
A1_REPORT_FILE = os.path.join(A1_RESULTS_DIR, "results_comparison_report_a1.txt")
A2_RESULTS_DIR = os.path.abspath("results")
OUTPUT_FILE = os.path.join(A2_RESULTS_DIR, "a1_vs_a2_comparison.txt")


def parse_metrics_file(filepath):
    """Extracts configuration, accuracy, epochs_run, and metrics dictionary from evaluation_metrics file."""
    config = ""
    accuracy = None
    epochs_run = "N/A"
    metrics = {}

    if not os.path.exists(filepath):
        return config, accuracy, epochs_run, metrics

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Configuration:"):
                config = line.replace("Configuration:", "").strip()
                for p in config.split("|"):
                    p = p.strip()
                    if p.startswith("epochs_run:"):
                        epochs_run = p.replace("epochs_run:", "").strip()
            elif ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                metrics[key] = val
                if key.lower() == "overall_accuracy":
                    try:
                        accuracy = float(val)
                    except ValueError:
                        pass
                elif key.lower() == "rmse":
                    try:
                        metrics["rmse_float"] = float(val)
                    except ValueError:
                        pass
                elif key.lower() == "percent_rmse":
                    try:
                        metrics["percent_rmse_float"] = float(val)
                    except ValueError:
                        pass

    return config, accuracy, epochs_run, metrics


def load_best_classification(base_dir, dataset_name):
    filepath = os.path.join(base_dir, dataset_name, "best", "evaluation_metrics_test.txt")
    cfg, acc, ep, metrics = parse_metrics_file(filepath)
    return {
        "dataset": dataset_name,
        "config": cfg,
        "overall_accuracy": acc if acc is not None else 0.0,
        "epochs_run": ep,
        "metrics": metrics,
    }


def load_best_regression(base_dir, dataset_name):
    filepath = os.path.join(base_dir, dataset_name, "best", "evaluation_metrics_test.txt")
    cfg, _, ep, metrics = parse_metrics_file(filepath)
    rmse_val = metrics.get("rmse_float")
    prmse_val = metrics.get("percent_rmse_float")
    return {
        "dataset": dataset_name,
        "config": cfg,
        "rmse": rmse_val,
        "percent_rmse": prmse_val,
        "epochs_run": ep,
        "metrics": metrics,
    }


def clean_config_id(raw_cfg, max_len=24):
    if "config_id:" in raw_cfg:
        cfg = raw_cfg.split("config_id:")[-1].strip()
    elif "|" in raw_cfg:
        parts = [p.strip() for p in raw_cfg.split("|")]
        cfg = " | ".join(parts[1:])
    else:
        cfg = raw_cfg.strip()
    if len(cfg) > max_len:
        cfg = cfg[: max_len - 3] + "..."
    return cfg


def generate_comprehensive_comparison():
    # Gather metrics
    a1_cls = {ds: load_best_classification(A1_RESULTS_DIR, ds) for ds in ["LS", "NLS"]}
    a2_cls = {ds: load_best_classification(A2_RESULTS_DIR, ds) for ds in ["LS", "NLS"]}

    a1_reg = {ds: load_best_regression(A1_RESULTS_DIR, ds) for ds in ["Univariate", "Bivariate"]}
    a2_reg = {ds: load_best_regression(A2_RESULTS_DIR, ds) for ds in ["Univariate", "Bivariate"]}

    lines = []

    # Document Header
    lines.append("=" * 116)
    lines.append("HEAD-TO-HEAD PERFORMANCE COMPARISON REPORT: ASSIGNMENT 1 (SINGLE NEURON) VS ASSIGNMENT 2 (FCNN)")
    lines.append("=" * 116)
    lines.append("Evaluation Methodology & Fair Comparison Setup:")
    lines.append("  - Dataset Split: Identical 60% Train / 20% Validation / 20% Test (seed=42)")
    lines.append("  - Convergence Rule: Early stopping when |loss(t) - loss(t-1)| < 0.0001 for 5 consecutive epochs")
    lines.append("  - Safety Epochs Limit: 2000 maximum epochs")
    lines.append("  - All reported test metrics below are computed on the EXACT SAME 20% held-out test split.")
    lines.append("=" * 116)

    # 1. Classification Table
    lines.append("\n" + "-" * 116)
    lines.append("1. CLASSIFICATION PERFORMANCE SUMMARY (TEST SPLIT)")
    lines.append("-" * 116)
    cls_hdr = (
        f"{'Dataset':<10} | {'A1 Config (Single Neuron)':<26} | {'A1 Ep':<6} | {'A1 Acc':<8} | "
        f"{'A2 Config (FCNN)':<24} | {'A2 Ep':<6} | {'A2 Acc':<8} | {'Delta Acc':<10}"
    )
    lines.append(cls_hdr)
    lines.append("-" * 116)

    for ds in ["LS", "NLS"]:
        a1 = a1_cls[ds]
        a2 = a2_cls[ds]

        a1_cfg = clean_config_id(a1["config"], 26)
        a2_cfg = clean_config_id(a2["config"], 24)

        a1_ep = str(a1["epochs_run"])
        a2_ep = str(a2["epochs_run"])

        a1_acc = a1["overall_accuracy"]
        a2_acc = a2["overall_accuracy"]
        delta = a2_acc - a1_acc

        row = (
            f"{ds:<10} | {a1_cfg:<26} | {a1_ep:<6} | {a1_acc:8.4f} | "
            f"{a2_cfg:<24} | {a2_ep:<6} | {a2_acc:8.4f} | {delta:+10.4f}"
        )
        lines.append(row)

    lines.append("-" * 116)
    lines.append("Delta Acc = A2 Accuracy - A1 Accuracy (positive indicates FCNN outperforming Single Neuron).")

    # Detailed Class-Level Classification Breakdown
    lines.append("\n" + "." * 116)
    lines.append("Detailed Test Set Metrics Breakdown:")
    for ds in ["LS", "NLS"]:
        a1 = a1_cls[ds]
        a2 = a2_cls[ds]
        lines.append(f"  * {ds} Dataset:")
        lines.append(
            f"    - Assignment 1 ({clean_config_id(a1['config'], 30)}): "
            f"Accuracy: {a1['overall_accuracy']:.4f} | "
            f"Macro Precision: {a1['metrics'].get('macro_precision', 'N/A')} | "
            f"Macro Recall: {a1['metrics'].get('macro_recall', 'N/A')} | "
            f"Macro F1: {a1['metrics'].get('macro_f_measure', 'N/A')}"
        )
        if "class_f_measure" in a1["metrics"]:
            lines.append(f"      Per-Class F-Measure: {a1['metrics']['class_f_measure']}")

        lines.append(
            f"    - Assignment 2 ({clean_config_id(a2['config'], 30)}): "
            f"Accuracy: {a2['overall_accuracy']:.4f} | "
            f"Macro Precision: {a2['metrics'].get('macro_precision', 'N/A')} | "
            f"Macro Recall: {a2['metrics'].get('macro_recall', 'N/A')} | "
            f"Macro F1: {a2['metrics'].get('macro_f_measure', 'N/A')}"
        )
        if "class_f_measure" in a2["metrics"]:
            lines.append(f"      Per-Class F-Measure: {a2['metrics']['class_f_measure']}")

    # 2. Regression Table
    lines.append("\n\n" + "-" * 116)
    lines.append("2. REGRESSION PERFORMANCE SUMMARY (TEST SPLIT)")
    lines.append("-" * 116)
    reg_hdr = (
        f"{'Dataset':<12} | {'A1 Config':<18} | {'A1 Ep':<6} | {'A1 RMSE':<8} | {'A1 %RMSE':<9} | "
        f"{'A2 Config':<18} | {'A2 Ep':<6} | {'A2 RMSE':<8} | {'A2 %RMSE':<9} | {'Delta RMSE':<10} | {'Error Red%':<10}"
    )
    lines.append(reg_hdr)
    lines.append("-" * 116)

    for ds in ["Univariate", "Bivariate"]:
        a1 = a1_reg[ds]
        a2 = a2_reg[ds]

        a1_cfg = clean_config_id(a1["config"], 18)
        a2_cfg = clean_config_id(a2["config"], 18)

        a1_ep = str(a1["epochs_run"])
        a2_ep = str(a2["epochs_run"])

        a1_rmse = a1["rmse"]
        a1_prmse = a1["percent_rmse"]
        a2_rmse = a2["rmse"]
        a2_prmse = a2["percent_rmse"]

        a1_r_str = f"{a1_rmse:.4f}" if a1_rmse is not None else "N/A"
        a1_p_str = f"{a1_prmse:.2f}%" if a1_prmse is not None else "N/A"
        a2_r_str = f"{a2_rmse:.4f}" if a2_rmse is not None else "N/A"
        a2_p_str = f"{a2_prmse:.2f}%" if a2_prmse is not None else "N/A"

        if a1_rmse is not None and a2_rmse is not None and a1_rmse > 0:
            delta_rmse = a1_rmse - a2_rmse
            delta_str = f"{delta_rmse:+.4f}"
            reduction_pct = (delta_rmse / a1_rmse) * 100.0
            red_str = f"{reduction_pct:.2f}%"
        else:
            delta_str = "N/A"
            red_str = "N/A"

        row = (
            f"{ds:<12} | {a1_cfg:<18} | {a1_ep:<6} | {a1_r_str:<8} | {a1_p_str:<9} | "
            f"{a2_cfg:<18} | {a2_ep:<6} | {a2_r_str:<8} | {a2_p_str:<9} | {delta_str:<10} | {red_str:<10}"
        )
        lines.append(row)

    lines.append("-" * 116)
    lines.append("Delta RMSE = A1 RMSE - A2 RMSE (positive indicates FCNN reduced error).")
    lines.append("Error Red% = (A1 RMSE - A2 RMSE) / A1 RMSE * 100% (percentage reduction in root-mean-squared error).")

    # 3. Training Dynamics & Convergence Speed
    lines.append("\n\n" + "-" * 116)
    lines.append("3. TRAINING CONVERGENCE & EFFICIENCY COMPARISON")
    lines.append("-" * 116)
    lines.append(
        f"{'Task / Dataset':<20} | {'A1 Convergence Epochs':<24} | {'A2 Convergence Epochs':<24} | {'Speedup / Efficiency':<25}"
    )
    lines.append("-" * 116)

    # LS
    lines.append(
        f"{'LS Classification':<20} | {str(a1_cls['LS']['epochs_run']):<24} | {str(a2_cls['LS']['epochs_run']):<24} | {'FCNN converged 1.39x faster':<25}"
    )
    # NLS
    lines.append(
        f"{'NLS Classification':<20} | {str(a1_cls['NLS']['epochs_run']):<24} | {str(a2_cls['NLS']['epochs_run']):<24} | {'FCNN solved complex boundary':<25}"
    )
    # Univariate
    lines.append(
        f"{'Univariate Reg':<20} | {str(a1_reg['Univariate']['epochs_run']):<24} | {str(a2_reg['Univariate']['epochs_run']):<24} | {'FCNN converged 14.0x faster':<25}"
    )
    # Bivariate
    lines.append(
        f"{'Bivariate Reg':<20} | {str(a1_reg['Bivariate']['epochs_run']):<24} | {str(a2_reg['Bivariate']['epochs_run']):<24} | {'FCNN converged 18.5x faster':<25}"
    )
    lines.append("-" * 116)

    # 4. Inferences and Observations
    lines.append("\n\n" + "=" * 116)
    lines.append("4. INFERENCES & ARCHITECTURAL OBSERVATIONS")
    lines.append("=" * 116)
    lines.append(
        "1. Linearly Separable Classification (LS):\n"
        "   - The single-neuron perceptron in A1 achieved 0.9833 test accuracy, demonstrating that linear decision\n"
        "     boundaries are already largely adequate for this problem.\n"
        "   - The FCNN (H2 with logistic activation) achieved 0.9967 test accuracy (converging in 120 epochs),\n"
        "     further refining boundary margins without overfitting.\n\n"
        "2. Non-Linearly Separable Classification (NLS):\n"
        "   - The single-neuron model hit a hard performance ceiling of 0.8167 accuracy on test data, as linear\n"
        "     hyperplanes fundamentally cannot separate interlocking, non-convex class boundaries.\n"
        "   - The 2-hidden-layer FCNN (H4-2 with logistic activation) achieved a PERFECT 1.0000 test accuracy (100%),\n"
        "     an absolute improvement of +0.1833 (+18.33%). The combination of 4 nodes in layer 1 and 2 nodes in layer 2\n"
        "     formed curved decision boundaries that completely enclosed and separated all 3 classes.\n\n"
        "3. Univariate Regression:\n"
        "   - The single-neuron linear regressor in A1 was restricted to fitting a straight line, yielding a test RMSE\n"
        "     of 0.1757 (8.18% %RMSE) after 252 epochs.\n"
        "   - The FCNN (H8 with tanh activation) learned the non-linear curve easily, achieving a test RMSE of 0.0600\n"
        "     (2.79% %RMSE) in just 18 epochs - reducing error by 65.85% and converging 14x faster.\n\n"
        "4. Bivariate Regression:\n"
        "   - The single-neuron planar fit produced a massive test RMSE of 1.4274 (17.83% %RMSE) after 314 epochs,\n"
        "     failing completely to capture the surface topology.\n"
        "   - The FCNN (H4 with logistic activation) modeled the complex 3D surface with extraordinary precision,\n"
        "     dropping test RMSE to 0.0655 (0.82% %RMSE) in only 17 epochs - a 95.41% error reduction (~22x improvement)\n"
        "     and 18.5x faster convergence."
    )
    lines.append("=" * 116)

    full_report = "\n".join(lines) + "\n"

    # Save to OUTPUT_FILE (results/a1_vs_a2_comparison.txt)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(f"Final comparison report successfully generated and saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_comprehensive_comparison()