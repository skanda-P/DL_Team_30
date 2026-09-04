import os
import glob


A1_RESULTS_DIR = os.path.abspath(os.path.join("..", "..", "assignment_1", "src", "results"))
A2_RESULTS_DIR = os.path.abspath("results")
OUTPUT_FILE = os.path.join(A2_RESULTS_DIR, "a1_vs_a2_comparison.txt")


def parse_metrics_file(filepath):
    # Extracts configuration and overall accuracy from evaluation_metrics file
    config = ""
    accuracy = None
    metrics = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Configuration:"):
                config = line.replace("Configuration:", "").strip()
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

    return config, accuracy, metrics


def load_a1_metrics(a1_results_dir, dataset_name):
    # Finds the best performing Assignment 1 model for the dataset across sweeps
    pattern = os.path.join(a1_results_dir, dataset_name, "*", "evaluation_metrics.txt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No Assignment 1 metrics found for {dataset_name} in {a1_results_dir}. "
            f"Please run classification.py in assignment_1/src first."
        )

    best_config = ""
    best_acc = -1.0
    best_metrics = {}

    for fp in files:
        cfg, acc, metrics = parse_metrics_file(fp)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_config = cfg
            best_metrics = metrics

    return {
        "dataset": dataset_name,
        "config": best_config,
        "overall_accuracy": best_acc,
        "metrics": best_metrics,
    }


def load_a2_best_metrics(a2_results_dir, dataset_name):
    # Reads the best test metrics for the dataset produced by Assignment 2
    filepath = os.path.join(a2_results_dir, dataset_name, "best", "evaluation_metrics_test.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Assignment 2 test metrics not found at {filepath}. "
            f"Please run classification.py first."
        )

    cfg, acc, metrics = parse_metrics_file(filepath)
    return {
        "dataset": dataset_name,
        "config": cfg,
        "overall_accuracy": acc if acc is not None else 0.0,
        "metrics": metrics,
    }


def write_comparison_table(a1_results, a2_results, out_path):
    # Generates a comparison table comparing single-neuron vs FCNN performance
    lines = []
    lines.append("=" * 96)
    lines.append("PERFORMANCE COMPARISON: ASSIGNMENT 1 (SINGLE NEURON) VS ASSIGNMENT 2 (FCNN)")
    lines.append("=" * 96)
    header = f"{'Dataset':<8} | {'A1 Config':<30} | {'A1 Acc':<8} | {'A2 Config':<30} | {'A2 Acc':<8} | {'Delta':<8}"
    lines.append(header)
    lines.append("-" * 96)

    for ds in ["LS", "NLS"]:
        a1 = a1_results.get(ds, {})
        a2 = a2_results.get(ds, {})

        a1_acc = a1.get("overall_accuracy", 0.0)
        a2_acc = a2.get("overall_accuracy", 0.0)
        delta = (a2_acc - a1_acc) * 100

        a1_cfg = a1.get("config", "N/A")
        # Keep config display compact
        if "|" in a1_cfg:
            parts = [p.strip() for p in a1_cfg.split("|")]
            a1_cfg = " | ".join(parts[1:])
        if len(a1_cfg) > 30:
            a1_cfg = a1_cfg[:27] + "..."

        a2_cfg = a2.get("config", "N/A")
        if "config_id:" in a2_cfg:
            a2_cfg = a2_cfg.split("config_id:")[-1].strip()
        elif "|" in a2_cfg:
            parts = [p.strip() for p in a2_cfg.split("|")]
            a2_cfg = " | ".join(parts[1:])
        if len(a2_cfg) > 30:
            a2_cfg = a2_cfg[:27] + "..."

        row = (
            f"{ds:<8} | {a1_cfg:<30} | {a1_acc * 100:6.2f}% | "
            f"{a2_cfg:<30} | {a2_acc * 100:6.2f}% | {delta:+6.2f}%"
        )
        lines.append(row)

    lines.append("=" * 96)
    lines.append(
        "Caveat: Assignment 1 evaluated on a 70/30 train/test split, whereas Assignment 2 used\n"
        "a 60/20/20 train/validation/test split. Test sets are not identical, though drawn\n"
        "from the same underlying dataset."
    )

    table_text = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_text)

    print(table_text)
    print(f"Saved comparison to: {out_path}")


def main():
    datasets = ["LS", "NLS"]
    a1_metrics = {}
    a2_metrics = {}

    for ds in datasets:
        a1_metrics[ds] = load_a1_metrics(A1_RESULTS_DIR, ds)
        a2_metrics[ds] = load_a2_best_metrics(A2_RESULTS_DIR, ds)

    write_comparison_table(a1_metrics, a2_metrics, OUTPUT_FILE)


if __name__ == "__main__":
    main()
