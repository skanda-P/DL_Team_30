import os
import glob


def generate_comparison_report(results_dir="results", out_file="results_comparison_report.txt"):
    pattern = os.path.join(results_dir, "**", "evaluation_metrics*.txt")
    filepaths = glob.glob(pattern, recursive=True)

    if not filepaths:
        print(f"No results found in {results_dir}. Run classification.py first.")
        return

    lines = []
    lines.append("=" * 100)
    lines.append("ASSIGNMENT 2 ARCHITECTURE EVALUATION REPORT")
    lines.append("=" * 100)

    for ds in ["LS", "NLS"]:
        lines.append(f"\n--- Dataset: {ds} ---")
        sweep_pattern = os.path.join(results_dir, ds, "sweep", "*", "evaluation_metrics.txt")
        sweep_files = sorted(glob.glob(sweep_pattern))

        lines.append(f"{'Config ID':<35} | {'Val Accuracy':<15} | {'Epochs Run':<12}")
        lines.append("-" * 68)

        for fp in sweep_files:
            cfg_id = os.path.basename(os.path.dirname(fp))
            acc = "N/A"
            epochs_run = "N/A"
            with open(fp, "r") as f:
                for line in f:
                    if line.startswith("Configuration:"):
                        for part in line.split("|"):
                            if "epochs_run:" in part:
                                epochs_run = part.replace("epochs_run:", "").strip()
                    elif line.lower().startswith("overall_accuracy:"):
                        try:
                            val = float(line.split(":")[1].strip())
                            acc = f"{val * 100:.2f}%"
                        except ValueError:
                            pass
            lines.append(f"{cfg_id:<35} | {acc:<15} | {epochs_run:<12}")

        best_test_file = os.path.join(results_dir, ds, "best", "evaluation_metrics_test.txt")
        if os.path.exists(best_test_file):
            lines.append("-" * 68)
            with open(best_test_file, "r") as f:
                for line in f:
                    if line.lower().startswith("overall_accuracy:"):
                        try:
                            val = float(line.split(":")[1].strip())
                            lines.append(f"BEST MODEL TEST ACCURACY: {val * 100:.2f}%")
                        except ValueError:
                            pass

    lines.append("\n" + "=" * 100)
    report_text = "\n".join(lines) + "\n"

    with open(out_file, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"Report saved to: {out_file}")


def generate_regression_comparison_report(results_dir="results", out_file="results_comparison_report.txt"):
    # RMSE-based counterpart to generate_comparison_report(), for the
    # Univariate / Bivariate regression datasets produced by regression.py.
    # Appends to the same out_file so both reports live in one place.
    pattern = os.path.join(results_dir, "**", "evaluation_metrics*.txt")
    filepaths = glob.glob(pattern, recursive=True)

    if not filepaths:
        print(f"No results found in {results_dir}. Run regression.py first.")
        return

    lines = []
    lines.append("=" * 100)
    lines.append("ASSIGNMENT 2 REGRESSION ARCHITECTURE EVALUATION REPORT")
    lines.append("=" * 100)

    for ds in ["Univariate", "Bivariate"]:
        lines.append(f"\n--- Dataset: {ds} ---")
        sweep_pattern = os.path.join(results_dir, ds, "sweep", "*", "evaluation_metrics.txt")
        sweep_files = sorted(glob.glob(sweep_pattern))

        lines.append(
            f"{'Config ID':<35} | {'Train RMSE':<11} | {'Train %RMSE':<12} | "
            f"{'Val RMSE':<9} | {'Val %RMSE':<10} | {'Epochs Run':<12}"
        )
        lines.append("-" * 100)

        for fp in sweep_files:
            cfg_id = os.path.basename(os.path.dirname(fp))
            train_rmse = "N/A"
            train_percent_rmse = "N/A"
            val_rmse = "N/A"
            val_percent_rmse = "N/A"
            epochs_run = "N/A"
            with open(fp, "r") as f:
                for line in f:
                    if line.startswith("Configuration:"):
                        for part in line.split("|"):
                            if "epochs_run:" in part:
                                epochs_run = part.replace("epochs_run:", "").strip()
                    elif line.lower().startswith("train_rmse:"):
                        try:
                            val = float(line.split(":", 1)[1].strip())
                            train_rmse = f"{val:.4f}"
                        except ValueError:
                            pass
                    elif line.lower().startswith("train_percent_rmse:"):
                        try:
                            val = float(line.split(":", 1)[1].strip())
                            train_percent_rmse = f"{val:.2f}%"
                        except ValueError:
                            pass
                    elif line.lower().startswith("rmse:"):
                        try:
                            val = float(line.split(":", 1)[1].strip())
                            val_rmse = f"{val:.4f}"
                        except ValueError:
                            pass
                    elif line.lower().startswith("percent_rmse:"):
                        try:
                            val = float(line.split(":", 1)[1].strip())
                            val_percent_rmse = f"{val:.2f}%"
                        except ValueError:
                            pass
            lines.append(
                f"{cfg_id:<35} | {train_rmse:<11} | {train_percent_rmse:<12} | "
                f"{val_rmse:<9} | {val_percent_rmse:<10} | {epochs_run:<12}"
            )

        best_test_file = os.path.join(results_dir, ds, "best", "evaluation_metrics_test.txt")
        if os.path.exists(best_test_file):
            lines.append("-" * 100)
            best_rmse, best_percent_rmse = None, None
            with open(best_test_file, "r") as f:
                for line in f:
                    if line.lower().startswith("rmse:"):
                        try:
                            best_rmse = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    elif line.lower().startswith("percent_rmse:"):
                        try:
                            best_percent_rmse = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
            if best_rmse is not None:
                lines.append(
                    f"BEST MODEL TEST RMSE: {best_rmse:.4f}"
                    + (f"  (%RMSE: {best_percent_rmse:.2f}%)" if best_percent_rmse is not None else "")
                )

    lines.append("\n" + "=" * 100)
    report_text = "\n".join(lines) + "\n"

    # Append after the classification report so everything lands in one file.
    with open(out_file, "a") as f:
        f.write(report_text)

    print(report_text)
    print(f"Report appended to: {out_file}")


if __name__ == "__main__":
    generate_comparison_report()
    generate_regression_comparison_report()