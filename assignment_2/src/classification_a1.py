import os
import numpy as np

from multiclass import OneVsOneClassifier
from utils.data_utils import load_LS_data, load_nls_data, stratified_three_way_split
from utils.metrics import classification_metrics
from utils.plotting import plot_decision_regions, plot_error_vs_epochs


DATA_DIR = "data"
LS_DIR = os.path.join(DATA_DIR, "Classification", "LS_Group30")
NLS_FILE = os.path.join(DATA_DIR, "Classification", "NLS_Group30.txt")

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
SEED = 42

ACTIVATIONS = ["logistic", "tanh"]
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.2]
MAX_EPOCHS = 2000
STOPPING_THRESHOLD = 0.0001
PATIENCE = 5
SELECTION_METRIC = "overall_accuracy"


def save_metrics_file(filepath, dataset_name, cfg, metrics, epochs_run):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(
            f"Configuration: {dataset_name} | {cfg['activation']} | "
            f"LR: {cfg['learning_rate']} | EP: {cfg['epochs']} | "
            f"epochs_run: {epochs_run} | config_id: {cfg['config_id']}\n"
        )
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if isinstance(value, (float, np.floating)):
                f.write(f"{key}: {value:.4f}\n")
            elif isinstance(value, np.ndarray) and value.dtype.kind == "f":
                formatted_arr = "[" + " ".join(f"{v:.4f}" for v in value) + "]"
                f.write(f"{key}: {formatted_arr}\n")
            else:
                f.write(f"{key}: {value}\n")


def run_dataset(dataset_name, X, y):
    print(f"\n==========================================")
    print(f"Running A1 Classification: {dataset_name}")
    print(f"==========================================")

    # Identical 60/20/20 data split as Assignment 2
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(
        X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
    )

    all_classes = np.unique(y)
    num_classes = len(all_classes)

    sweep_results = []

    for act in ACTIVATIONS:
        for lr in LEARNING_RATES:
            cfg_id = f"{act}_LR{lr}"
            cfg = {
                "activation": act,
                "learning_rate": lr,
                "epochs": MAX_EPOCHS,
                "config_id": cfg_id,
            }

            clf = OneVsOneClassifier(
                activation=act,
                learning_rate=lr,
                epochs=MAX_EPOCHS,
                stopping_threshold=STOPPING_THRESHOLD,
                patience=PATIENCE,
            )
            clf.fit(X_train, y_train)

            # Max epochs run among the pairwise classifiers
            epochs_run = max(m.epochs_run for m in clf.classifiers.values())

            y_val_pred = clf.predict(X_val)
            val_metrics = classification_metrics(y_val, y_val_pred, num_classes)

            sweep_results.append({
                "config": cfg,
                "model": clf,
                "val_metrics": val_metrics,
                "epochs_run": epochs_run,
            })

            # Save candidate metrics in results/a1
            cand_dir = os.path.join("results", "a1", dataset_name, cfg_id)
            save_metrics_file(
                os.path.join(cand_dir, "evaluation_metrics.txt"),
                dataset_name, cfg, val_metrics, epochs_run,
            )

    # Select best model on validation split
    best = max(sweep_results, key=lambda entry: entry["val_metrics"][SELECTION_METRIC])
    best_cfg = best["config"]
    best_clf = best["model"]
    best_val_metrics = best["val_metrics"]

    # Evaluate best on train and identical test set
    y_train_pred = best_clf.predict(X_train)
    best_train_metrics = classification_metrics(y_train, y_train_pred, num_classes)
    y_test_pred = best_clf.predict(X_test)
    best_test_metrics = classification_metrics(y_test, y_test_pred, num_classes)

    best_dir = os.path.join("results", "a1", dataset_name, "best")
    os.makedirs(best_dir, exist_ok=True)

    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_train.txt"),
        dataset_name, best_cfg, best_train_metrics, best["epochs_run"],
    )
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_val.txt"),
        dataset_name, best_cfg, best_val_metrics, best["epochs_run"],
    )
    save_metrics_file(
        os.path.join(best_dir, "evaluation_metrics_test.txt"),
        dataset_name, best_cfg, best_test_metrics, best["epochs_run"],
    )

    # Plot error curves and decision boundary for best A1 model
    for (class_a, class_b), model in best_clf.classifiers.items():
        plot_error_vs_epochs(
            model.errors,
            title=f"A1 {dataset_name} ({best_cfg['activation']}): Error (Class {class_a} vs {class_b})",
            filename=os.path.join(best_dir, f"error_class{class_a}_vs_class{class_b}.png"),
        )

    plot_decision_regions(
        X_train, y_train, best_clf.predict,
        title=f"A1 {dataset_name} ({best_cfg['config_id']}): Decision Region",
        filename=os.path.join(best_dir, "decision_region.png"),
        all_classes=all_classes,
    )

    print(f"Best A1 Config: {best_cfg['config_id']}")
    print(f"Val Accuracy:  {best_val_metrics['overall_accuracy']:.4f}")
    print(f"Test Accuracy: {best_test_metrics['overall_accuracy']:.4f} (Epochs: {best['epochs_run']})")

    return best


def main():
    X_ls, y_ls = load_LS_data(LS_DIR)
    X_nls, y_nls = load_nls_data(NLS_FILE)

    run_dataset("LS", X_ls, y_ls)
    run_dataset("NLS", X_nls, y_nls)


if __name__ == "__main__":
    main()
