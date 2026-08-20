import os
import glob

def generate_comparison_report():

    search_pattern = os.path.join("results", "**", "evaluation_metrics.txt")
    filepaths = glob.glob(search_pattern, recursive=True)

    if not filepaths:
        print("No results found. Run your main_classification.py and main_regression.py scripts first!")
        return


    class_results = {
        "LS": {"logistic": [], "tanh": []},
        "NLS": {"logistic": [], "tanh": []}
    }
    reg_results = {
        "Univariate": [],
        "Bivariate": []
    }

    for path in filepaths:
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        dataset = activation = lr = epochs = ""
        accuracy = test_rmse = test_prmse = "N/A"
        is_classification = False

        for line in lines:

            if line.startswith("Configuration:"):
                parts = line.replace("Configuration:", "").split("|")
                if len(parts) >= 4:
                    dataset = parts[0].strip()
                    activation = parts[1].strip()
                    lr = parts[2].replace("LR:", "").strip()
                    epochs = parts[3].replace("EP:", "").strip()

                    if activation in ["logistic", "tanh"]:
                        is_classification = True


            elif line.lower().startswith("accuracy"):
                accuracy = line.split(":")[1].strip()
            elif line.startswith("Test RMSE:"):
                rmse_parts = line.split("|")
                test_rmse = rmse_parts[0].replace("Test RMSE:", "").strip()
                test_prmse = rmse_parts[1].replace("Test %RMSE:", "").strip()


        config_str = f"LR: {lr:<6} | Epochs: {epochs:<6}"

        if is_classification and dataset in class_results:
            class_results[dataset][activation].append(f"{config_str} | Accuracy: {accuracy}")
        elif not is_classification and dataset in reg_results:
            reg_results[dataset].append(f"{config_str} | Test RMSE: {test_rmse} | Test %RMSE: {test_prmse}")


    output_file = "results_comparison_report.txt"
    with open(output_file, "w") as f:
        f.write("=== HYPERPARAMETER COMPARISON REPORT ===\n")


        f.write("\n" + "="*40 + "\n")
        f.write("1. LS CLASSIFICATION\n")
        f.write("="*40 + "\n")
        for act in ["logistic", "tanh"]:
            f.write(f"\n--- Activation: {act.upper()} ---\n")
            for result in sorted(class_results["LS"][act]):
                f.write(f"  {result}\n")


        f.write("\n\n" + "="*40 + "\n")
        f.write("2. NLS CLASSIFICATION\n")
        f.write("="*40 + "\n")
        for act in ["logistic", "tanh"]:
            f.write(f"\n--- Activation: {act.upper()} ---\n")
            for result in sorted(class_results["NLS"][act]):
                f.write(f"  {result}\n")


        f.write("\n\n" + "="*40 + "\n")
        f.write("3. REGRESSION: UNIVARIATE\n")
        f.write("="*40 + "\n\n")
        for result in sorted(reg_results["Univariate"]):
            f.write(f"  {result}\n")


        f.write("\n\n" + "="*40 + "\n")
        f.write("4. REGRESSION: BIVARIATE\n")
        f.write("="*40 + "\n\n")
        for result in sorted(reg_results["Bivariate"]):
            f.write(f"  {result}\n")

    print(f"Comparison report successfully generated: {output_file}")

if __name__ == "__main__":
    generate_comparison_report()
