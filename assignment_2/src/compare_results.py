import os
import glob
import shutil


def generate_comparison_report(results_dir="results", out_file=None):
    search_pattern = os.path.join(results_dir, "**", "evaluation_metrics*.txt")
    filepaths = glob.glob(search_pattern, recursive=True)

    if not filepaths:
        print(f"No results found in {results_dir}. Run your classification.py first!")
        return

    class_results = {
        "LS": {"logistic": [], "tanh": []},
        "NLS": {"logistic": [], "tanh": []}
    }
    reg_results = {
        "Univariate": {"logistic": [], "tanh": []},
        "Bivariate": {"logistic": [], "tanh": []}
    }
    best_results = {}

    for path in filepaths:
        # Ignore best folder and a1 folder here
        path_parts = path.replace("\\", "/").split("/")
        if os.path.basename(os.path.dirname(path)) == "best" or "a1" in path_parts:
            continue

        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        dataset = activation = lr = epochs = arch = epochs_run = ""
        accuracy = "N/A"
        val_rmse = "N/A"
        val_prmse = "N/A"
        in_val_section = False

        for line in lines:
            if line.startswith("Configuration:"):
                parts = line.replace("Configuration:", "").split("|")
                if len(parts) >= 4:
                    dataset = parts[0].strip()
                    activation = parts[1].strip()
                    lr = parts[2].replace("LR:", "").strip()
                    epochs = parts[3].replace("EP:", "").strip()

                for p in parts:
                    p = p.strip()
                    if p.startswith("epochs_run:"):
                        epochs_run = p.replace("epochs_run:", "").strip()
                    elif p.startswith("config_id:"):
                        cfg_id = p.replace("config_id:", "").strip()
                        arch = cfg_id.split(f"_{activation}_")[0] if f"_{activation}_" in cfg_id else cfg_id

            elif line.lower().startswith("overall_accuracy"):
                accuracy = line.split(":")[1].strip()
            elif line.strip() == "[validation]":
                in_val_section = True
            elif in_val_section and line.startswith("rmse:"):
                val_rmse = line.split(":")[1].strip()
            elif in_val_section and line.startswith("percent_rmse:"):
                val_prmse = line.split(":")[1].strip()
            elif line.startswith("Test RMSE:"):
                rmse_parts = line.split("|")
                val_rmse = rmse_parts[0].replace("Test RMSE:", "").strip()
                val_prmse = rmse_parts[1].replace("Test %RMSE:", "").strip()

        if arch:
            if epochs_run:
                config_str = f"Arch: {arch:<10} | LR: {lr:<6} | Epochs to Converge: {epochs_run:<4}"
            else:
                config_str = f"Arch: {arch:<10} | LR: {lr:<6}"
        else:
            if epochs_run:
                config_str = f"LR: {lr:<6} | Epochs to Converge: {epochs_run:<4}"
            else:
                config_str = f"LR: {lr:<6}"

        if dataset in class_results and activation in class_results[dataset]:
            acc_str = accuracy
            try:
                acc_str = f"{float(accuracy):.4f}"
            except (ValueError, TypeError):
                pass
            class_results[dataset][activation].append(f"{config_str} | Accuracy: {acc_str}")

        elif dataset in reg_results and activation in reg_results[dataset]:
            try:
                val_rmse_str = f"{float(val_rmse):.4f}"
            except (ValueError, TypeError):
                val_rmse_str = val_rmse
            try:
                val_prmse_str = f"{float(val_prmse):.2f}%"
            except (ValueError, TypeError):
                val_prmse_str = val_prmse
            reg_results[dataset][activation].append(
                f"{config_str} | Val RMSE: {val_rmse_str} | Val %RMSE: {val_prmse_str}"
            )

    # Check for best model test metrics across all datasets
    for ds in ["LS", "NLS", "Univariate", "Bivariate"]:
        test_f = os.path.join(results_dir, ds, "best", "evaluation_metrics_test.txt")
        if os.path.exists(test_f):
            test_acc = "N/A"
            test_rmse = "N/A"
            test_prmse = "N/A"
            cfg_name = "N/A"
            test_ep_run = ""
            with open(test_f) as f:
                for line in f:
                    if line.startswith("Configuration:"):
                        parts = line.split("|")
                        for p in parts:
                            p = p.strip()
                            if p.startswith("epochs_run:"):
                                test_ep_run = p.replace("epochs_run:", "").strip()
                    if "config_id:" in line:
                        cfg_name = line.split("config_id:")[-1].strip()
                    elif line.lower().startswith("overall_accuracy"):
                        raw_acc = line.split(":")[1].strip()
                        try:
                            test_acc = f"{float(raw_acc):.4f}"
                        except (ValueError, TypeError):
                            test_acc = raw_acc
                    elif line.startswith("rmse:"):
                        raw_rmse = line.split(":")[1].strip()
                        try:
                            test_rmse = f"{float(raw_rmse):.4f}"
                        except (ValueError, TypeError):
                            test_rmse = raw_rmse
                    elif line.startswith("percent_rmse:"):
                        raw_prmse = line.split(":")[1].strip()
                        try:
                            test_prmse = f"{float(raw_prmse):.2f}%"
                        except (ValueError, TypeError):
                            test_prmse = raw_prmse

            best_results[ds] = {
                "config": cfg_name,
                "test_accuracy": test_acc,
                "test_rmse": test_rmse,
                "test_prmse": test_prmse,
                "epochs_run": test_ep_run,
            }

    # Build report text matching Assignment 2 structure
    report_lines = []
    report_lines.append("=== HYPERPARAMETER COMPARISON REPORT (ASSIGNMENT 2 - FCNN) ===")

    # 1. LS Classification
    report_lines.append("\n" + "=" * 40)
    report_lines.append("1. LS CLASSIFICATION")
    report_lines.append("=" * 40)
    for act in ["logistic", "tanh"]:
        report_lines.append(f"\n--- Activation: {act.upper()} ---")
        for result in sorted(class_results["LS"][act]):
            report_lines.append(f"  {result}")
    if "LS" in best_results:
        report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
        report_lines.append(f"  Best Config: {best_results['LS']['config']}")
        if best_results['LS'].get('epochs_run'):
            report_lines.append(f"  Epochs to Converge: {best_results['LS']['epochs_run']}")
        report_lines.append(f"  Test Accuracy: {best_results['LS']['test_accuracy']}")

    # 2. NLS Classification
    report_lines.append("\n\n" + "=" * 40)
    report_lines.append("2. NLS CLASSIFICATION")
    report_lines.append("=" * 40)
    for act in ["logistic", "tanh"]:
        report_lines.append(f"\n--- Activation: {act.upper()} ---")
        for result in sorted(class_results["NLS"][act]):
            report_lines.append(f"  {result}")
    if "NLS" in best_results:
        report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
        report_lines.append(f"  Best Config: {best_results['NLS']['config']}")
        if best_results['NLS'].get('epochs_run'):
            report_lines.append(f"  Epochs to Converge: {best_results['NLS']['epochs_run']}")
        report_lines.append(f"  Test Accuracy: {best_results['NLS']['test_accuracy']}")

    # 3. Univariate Regression
    if any(reg_results["Univariate"].values()):
        report_lines.append("\n\n" + "=" * 40)
        report_lines.append("3. REGRESSION: UNIVARIATE")
        report_lines.append("=" * 40)
        for act in ["logistic", "tanh"]:
            if reg_results["Univariate"][act]:
                report_lines.append(f"\n--- Activation: {act.upper()} ---")
                for result in sorted(reg_results["Univariate"][act]):
                    report_lines.append(f"  {result}")
        if "Univariate" in best_results:
            report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
            report_lines.append(f"  Best Config: {best_results['Univariate']['config']}")
            if best_results['Univariate'].get('epochs_run'):
                report_lines.append(f"  Epochs to Converge: {best_results['Univariate']['epochs_run']}")
            report_lines.append(f"  Test RMSE: {best_results['Univariate']['test_rmse']} | Test %RMSE: {best_results['Univariate']['test_prmse']}")

    # 4. Bivariate Regression
    if any(reg_results["Bivariate"].values()):
        report_lines.append("\n\n" + "=" * 40)
        report_lines.append("4. REGRESSION: BIVARIATE")
        report_lines.append("=" * 40)
        for act in ["logistic", "tanh"]:
            if reg_results["Bivariate"][act]:
                report_lines.append(f"\n--- Activation: {act.upper()} ---")
                for result in sorted(reg_results["Bivariate"][act]):
                    report_lines.append(f"  {result}")
        if "Bivariate" in best_results:
            report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
            report_lines.append(f"  Best Config: {best_results['Bivariate']['config']}")
            if best_results['Bivariate'].get('epochs_run'):
                report_lines.append(f"  Epochs to Converge: {best_results['Bivariate']['epochs_run']}")
            report_lines.append(f"  Test RMSE: {best_results['Bivariate']['test_rmse']} | Test %RMSE: {best_results['Bivariate']['test_prmse']}")

    report_text = "\n".join(report_lines) + "\n"

    # Default output locations
    if out_file is None:
        out_file = os.path.join(results_dir, "summary.txt")

    os.makedirs(results_dir, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(report_text)

    # Synchronize to results/results_comparison_report.txt as well
    rep_path = os.path.join(results_dir, "results_comparison_report.txt")
    with open(rep_path, "w") as f:
        f.write(report_text)

    # For top-level compatibility if out_file is different
    root_rep = "results_comparison_report.txt"
    if os.path.abspath(out_file) != os.path.abspath(root_rep):
        try:
            with open(root_rep, "w") as f:
                f.write(report_text)
        except OSError:
            pass

    print(report_text)
    print(f"Comparison report successfully generated: {out_file}")
    print(f"Summary saved to: {rep_path}")


def generate_a1_report(a1_results_dir=None, out_file=None):
    if a1_results_dir is None:
        a1_results_dir = os.path.join("results", "a1")

    search_pattern = os.path.join(a1_results_dir, "**", "evaluation_metrics*.txt")
    filepaths = glob.glob(search_pattern, recursive=True)

    if not filepaths:
        print(f"No results found in {a1_results_dir}. Run your classification_a1.py and regression_a1.py first!")
        return

    class_results = {
        "LS": {"logistic": [], "tanh": []},
        "NLS": {"logistic": [], "tanh": []}
    }
    reg_results = {
        "Univariate": {"linear": []},
        "Bivariate": {"linear": []}
    }
    best_results = {}

    for path in filepaths:
        if os.path.basename(os.path.dirname(path)) == "best":
            continue

        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        dataset = activation = lr = epochs_run = ""
        accuracy = "N/A"
        val_rmse = "N/A"
        val_prmse = "N/A"

        for line in lines:
            if line.startswith("Configuration:"):
                parts = line.replace("Configuration:", "").split("|")
                if len(parts) >= 2:
                    dataset = parts[0].strip()
                    activation = parts[1].strip().lower()
                for p in parts:
                    p = p.strip()
                    if p.startswith("LR:"):
                        lr = p.replace("LR:", "").strip()
                    elif p.startswith("epochs_run:"):
                        epochs_run = p.replace("epochs_run:", "").strip()

            elif line.lower().startswith("overall_accuracy"):
                accuracy = line.split(":")[1].strip()
            elif line.startswith("rmse:") or line.startswith("Test RMSE:"):
                val_rmse = line.split(":")[1].strip().split("|")[0].strip()
            elif line.startswith("percent_rmse:") or line.startswith("Test %RMSE:"):
                val_prmse = line.split(":")[1].strip()

        config_str = f"LR: {lr:<6} | Epochs to Converge: {epochs_run:<4}"

        if dataset in class_results and activation in class_results[dataset]:
            try:
                acc_str = f"{float(accuracy):.4f}"
            except (ValueError, TypeError):
                acc_str = accuracy
            class_results[dataset][activation].append(f"{config_str} | Accuracy: {acc_str}")

        elif dataset in reg_results:
            if activation not in reg_results[dataset]:
                reg_results[dataset][activation] = []
            try:
                val_rmse_str = f"{float(val_rmse):.4f}"
            except (ValueError, TypeError):
                val_rmse_str = val_rmse
            try:
                val_prmse_str = f"{float(val_prmse):.2f}%"
            except (ValueError, TypeError):
                val_prmse_str = val_prmse
            reg_results[dataset][activation].append(
                f"{config_str} | Val RMSE: {val_rmse_str} | Val %RMSE: {val_prmse_str}"
            )

    # Check for best model test metrics across A1 datasets
    for ds in ["LS", "NLS", "Univariate", "Bivariate"]:
        test_f = os.path.join(a1_results_dir, ds, "best", "evaluation_metrics_test.txt")
        if os.path.exists(test_f):
            test_acc = "N/A"
            test_rmse = "N/A"
            test_prmse = "N/A"
            cfg_name = "N/A"
            test_ep_run = ""
            with open(test_f) as f:
                for line in f:
                    if line.startswith("Configuration:"):
                        parts = line.split("|")
                        for p in parts:
                            p = p.strip()
                            if p.startswith("epochs_run:"):
                                test_ep_run = p.replace("epochs_run:", "").strip()
                            elif p.startswith("config_id:"):
                                cfg_name = p.replace("config_id:", "").strip()
                    elif line.lower().startswith("overall_accuracy"):
                        raw_acc = line.split(":")[1].strip()
                        try:
                            test_acc = f"{float(raw_acc):.4f}"
                        except (ValueError, TypeError):
                            test_acc = raw_acc
                    elif line.startswith("rmse:"):
                        raw_rmse = line.split(":")[1].strip()
                        try:
                            test_rmse = f"{float(raw_rmse):.4f}"
                        except (ValueError, TypeError):
                            test_rmse = raw_rmse
                    elif line.startswith("percent_rmse:"):
                        raw_prmse = line.split(":")[1].strip()
                        try:
                            test_prmse = f"{float(raw_prmse):.2f}%"
                        except (ValueError, TypeError):
                            test_prmse = raw_prmse

            best_results[ds] = {
                "config": cfg_name,
                "test_accuracy": test_acc,
                "test_rmse": test_rmse,
                "test_prmse": test_prmse,
                "epochs_run": test_ep_run,
            }

    report_lines = []
    report_lines.append("=== HYPERPARAMETER COMPARISON REPORT (ASSIGNMENT 1 - SINGLE NEURON) ===")

    # 1. LS Classification
    report_lines.append("\n" + "=" * 40)
    report_lines.append("1. LS CLASSIFICATION")
    report_lines.append("=" * 40)
    for act in ["logistic", "tanh"]:
        report_lines.append(f"\n--- Activation: {act.upper()} ---")
        for result in sorted(class_results["LS"][act]):
            report_lines.append(f"  {result}")
    if "LS" in best_results:
        report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
        report_lines.append(f"  Best Config: {best_results['LS']['config']}")
        if best_results['LS'].get('epochs_run'):
            report_lines.append(f"  Epochs to Converge: {best_results['LS']['epochs_run']}")
        report_lines.append(f"  Test Accuracy: {best_results['LS']['test_accuracy']}")

    # 2. NLS Classification
    report_lines.append("\n\n" + "=" * 40)
    report_lines.append("2. NLS CLASSIFICATION")
    report_lines.append("=" * 40)
    for act in ["logistic", "tanh"]:
        report_lines.append(f"\n--- Activation: {act.upper()} ---")
        for result in sorted(class_results["NLS"][act]):
            report_lines.append(f"  {result}")
    if "NLS" in best_results:
        report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
        report_lines.append(f"  Best Config: {best_results['NLS']['config']}")
        if best_results['NLS'].get('epochs_run'):
            report_lines.append(f"  Epochs to Converge: {best_results['NLS']['epochs_run']}")
        report_lines.append(f"  Test Accuracy: {best_results['NLS']['test_accuracy']}")

    # 3. Univariate Regression
    if any(reg_results["Univariate"].values()):
        report_lines.append("\n\n" + "=" * 40)
        report_lines.append("3. REGRESSION: UNIVARIATE")
        report_lines.append("=" * 40)
        for act in sorted(reg_results["Univariate"].keys()):
            if reg_results["Univariate"][act]:
                report_lines.append(f"\n--- Activation: {act.upper()} ---")
                for result in sorted(reg_results["Univariate"][act]):
                    report_lines.append(f"  {result}")
        if "Univariate" in best_results:
            report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
            report_lines.append(f"  Best Config: {best_results['Univariate']['config']}")
            if best_results['Univariate'].get('epochs_run'):
                report_lines.append(f"  Epochs to Converge: {best_results['Univariate']['epochs_run']}")
            report_lines.append(f"  Test RMSE: {best_results['Univariate']['test_rmse']} | Test %RMSE: {best_results['Univariate']['test_prmse']}")

    # 4. Bivariate Regression
    if any(reg_results["Bivariate"].values()):
        report_lines.append("\n\n" + "=" * 40)
        report_lines.append("4. REGRESSION: BIVARIATE")
        report_lines.append("=" * 40)
        for act in sorted(reg_results["Bivariate"].keys()):
            if reg_results["Bivariate"][act]:
                report_lines.append(f"\n--- Activation: {act.upper()} ---")
                for result in sorted(reg_results["Bivariate"][act]):
                    report_lines.append(f"  {result}")
        if "Bivariate" in best_results:
            report_lines.append("\n--- Best Architecture (Cross-Validation) ---")
            report_lines.append(f"  Best Config: {best_results['Bivariate']['config']}")
            if best_results['Bivariate'].get('epochs_run'):
                report_lines.append(f"  Epochs to Converge: {best_results['Bivariate']['epochs_run']}")
            report_lines.append(f"  Test RMSE: {best_results['Bivariate']['test_rmse']} | Test %RMSE: {best_results['Bivariate']['test_prmse']}")

    report_text = "\n".join(report_lines) + "\n"

    if out_file is None:
        out_file = os.path.join(a1_results_dir, "results_comparison_report_a1.txt")

    os.makedirs(a1_results_dir, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(report_text)

    # Save to summary.txt inside results/a1/
    summary_path = os.path.join(a1_results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(report_text)

    # Also save into results/results_comparison_report_a1.txt so it's in results folder root
    res_root_file = os.path.join(os.path.dirname(a1_results_dir), "results_comparison_report_a1.txt")
    with open(res_root_file, "w") as f:
        f.write(report_text)

    # Remove any old results_comparison_report_a1.txt sitting in src/
    stale_src_file = "results_comparison_report_a1.txt"
    if os.path.exists(stale_src_file):
        try:
            os.remove(stale_src_file)
        except OSError:
            pass

    print(report_text)
    print(f"Assignment 1 report successfully generated: {out_file}")
    print(f"Also saved to: {summary_path} and {res_root_file}")


if __name__ == "__main__":
    generate_comparison_report()
    generate_a1_report()