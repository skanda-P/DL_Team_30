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


if __name__ == "__main__":
    generate_comparison_report()
