import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
A1_RESULTS_DIR = os.path.join(RESULTS_DIR, "a1")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "a1_vs_a2_comparison.txt")

CLASSIFICATION_DATASETS = ("LS", "NLS")
REGRESSION_DATASETS = ("Univariate", "Bivariate")


def parse_metrics_file(filepath):
    values = {}
    if not os.path.exists(filepath):
        return values

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Configuration:"):
                for part in line.split("|"):
                    key, separator, value = part.strip().partition(":")
                    if separator:
                        values[key] = value.strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()

    for key in ("overall_accuracy", "rmse", "percent_rmse"):
        try:
            values[f"{key}_float"] = float(values[key])
        except (KeyError, ValueError):
            values[f"{key}_float"] = None

    try:
        values["epochs_run_int"] = int(values["epochs_run"])
    except (KeyError, ValueError):
        values["epochs_run_int"] = None

    return values


def load_best(base_dir, dataset_name):
    filepath = os.path.join(base_dir, dataset_name, "best", "evaluation_metrics_test.txt")
    values = parse_metrics_file(filepath)
    values["dataset"] = dataset_name
    values["source"] = filepath
    return values


def format_value(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def format_epochs(value):
    return "N/A" if value is None else str(value)


def epoch_comparison(a1_epochs, a2_epochs):
    if a1_epochs is None or a2_epochs is None:
        return "N/A", "N/A"

    difference = a2_epochs - a1_epochs
    if difference < 0:
        result = f"A2 uses {abs(difference)} fewer"
    elif difference > 0:
        result = f"A1 uses {difference} fewer"
    else:
        result = "Equal"
    return str(difference), result


def append_classification_section(lines, results):
    lines.extend([
        "CLASSIFICATION TEST RESULTS",
        "Dataset | A1 Config | A1 Accuracy | A1 Epochs | A2 Config | A2 Accuracy | A2 Epochs | Epoch Delta (A2-A1) | Comparison",
        "--------|------------|--------------|-----------|------------|--------------|-----------|----------------------|-----------",
    ])

    for dataset, (a1, a2) in results.items():
        delta, comparison = epoch_comparison(
            a1.get("epochs_run_int"), a2.get("epochs_run_int")
        )
        lines.append(
            f"{dataset} | {a1.get('config_id', 'N/A')} | "
            f"{format_value(a1.get('overall_accuracy_float'))} | "
            f"{format_epochs(a1.get('epochs_run_int'))} | "
            f"{a2.get('config_id', 'N/A')} | "
            f"{format_value(a2.get('overall_accuracy_float'))} | "
            f"{format_epochs(a2.get('epochs_run_int'))} | {delta} | {comparison}"
        )


def append_regression_section(lines, results):
    lines.extend([
        "REGRESSION TEST RESULTS",
        "Dataset | A1 Config | A1 RMSE | A1 %RMSE | A1 Epochs | A2 Config | A2 RMSE | A2 %RMSE | A2 Epochs | Epoch Delta (A2-A1) | Comparison",
        "--------|------------|----------|----------|-----------|------------|----------|----------|-----------|----------------------|-----------",
    ])

    for dataset, (a1, a2) in results.items():
        delta, comparison = epoch_comparison(
            a1.get("epochs_run_int"), a2.get("epochs_run_int")
        )
        lines.append(
            f"{dataset} | {a1.get('config_id', 'N/A')} | "
            f"{format_value(a1.get('rmse_float'))} | "
            f"{format_value(a1.get('percent_rmse_float'), 2)}% | "
            f"{format_epochs(a1.get('epochs_run_int'))} | "
            f"{a2.get('config_id', 'N/A')} | "
            f"{format_value(a2.get('rmse_float'))} | "
            f"{format_value(a2.get('percent_rmse_float'), 2)}% | "
            f"{format_epochs(a2.get('epochs_run_int'))} | {delta} | {comparison}"
        )


def generate_comprehensive_comparison():
    classification = {
        dataset: (
            load_best(A1_RESULTS_DIR, dataset),
            load_best(RESULTS_DIR, dataset),
        )
        for dataset in CLASSIFICATION_DATASETS
    }
    regression = {
        dataset: (
            load_best(A1_RESULTS_DIR, dataset),
            load_best(RESULTS_DIR, dataset),
        )
        for dataset in REGRESSION_DATASETS
    }

    lines = [
        "ASSIGNMENT 1 VS ASSIGNMENT 2 COMPARISON",
        "",
        "The rows below use the saved best model for each dataset.",
        "Epoch Delta (A2-A1) is negative when Assignment 2 converged sooner.",
        "",
    ]
    append_classification_section(lines, classification)
    lines.append("")
    append_regression_section(lines, regression)
    lines.append("")

    report = "\n".join(lines) + "\n"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print(report)
    print(f"Comparison report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_comprehensive_comparison()
