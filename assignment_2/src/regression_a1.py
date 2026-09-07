import os
import numpy as np

from perceptron import Perceptron
from utils.data_utils import load_regression_csv, three_way_split
from utils.metrics import percent_rmse, rmse
from utils.plotting import (
    plot_error_vs_epochs, plot_regression_1d, plot_regression_2d, plot_target_vs_model_scatter
)
from utils.training_config import MAX_EPOCHS, PATIENCE, STOPPING_THRESHOLD


DATA_DIR = "data"
DATASETS = {
    "Univariate": {"path": os.path.join(DATA_DIR, "Regression", "UnivariateData", "30.csv"), "dim": 1},
    "Bivariate": {"path": os.path.join(DATA_DIR, "Regression", "BivariateData", "30.csv"), "dim": 2},
}

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
SEED = 42

LEARNING_RATES = [0.001, 0.01, 0.05, 0.1]


def save_metrics_file(filepath, dataset_name, cfg, metrics, epochs_run):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(
            f"Configuration: {dataset_name} | linear | "
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


def run_dataset(dataset_name, path, dim):
    print(f"\n==========================================")
    print(f"Running A1 Regression: {dataset_name}")
    print(f"==========================================")

    X, y = load_regression_csv(path)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
        X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
    )

    sweep_results = []

    for lr in LEARNING_RATES:
        cfg_id = f"linear_LR{lr}"
        cfg = {
            "learning_rate": lr,
            "epochs": MAX_EPOCHS,
            "config_id": cfg_id,
        }

        model = Perceptron(
            learning_rate=lr,
            epochs=MAX_EPOCHS,
            activation="linear",
            stopping_threshold=STOPPING_THRESHOLD,
            patience=PATIENCE,
        )
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_rmse = rmse(y_val, y_val_pred)
        val_prmse = percent_rmse(y_val, y_val_pred)
        val_metrics = {"rmse": val_rmse, "percent_rmse": val_prmse}

        sweep_results.append({
            "config": cfg,
            "model": model,
            "val_metrics": val_metrics,
            "epochs_run": model.epochs_run,
        })

        cand_dir = os.path.join("results", "a1", dataset_name, cfg_id)
        save_metrics_file(
            os.path.join(cand_dir, "evaluation_metrics.txt"),
            dataset_name, cfg, val_metrics, model.epochs_run,
        )

    # Select best model on validation split (lowest RMSE)
    best = min(
        sweep_results,
        key=lambda entry: (
            entry["val_metrics"]["rmse"],
            entry["epochs_run"],
        ),
    )
    best_cfg = best["config"]
    best_model = best["model"]
    best_val_metrics = best["val_metrics"]

    # Evaluate best on identical test set
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    test_rmse = rmse(y_test, y_test_pred)
    test_prmse = percent_rmse(y_test, y_test_pred)
    best_test_metrics = {"rmse": test_rmse, "percent_rmse": test_prmse}

    train_rmse_val = rmse(y_train, y_train_pred)
    train_prmse_val = percent_rmse(y_train, y_train_pred)
    best_train_metrics = {"rmse": train_rmse_val, "percent_rmse": train_prmse_val}

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

    # Plot error vs epochs
    plot_error_vs_epochs(
        best_model.errors,
        title=f"A1 {dataset_name} ({best_cfg['config_id']}): Error vs Epochs",
        filename=os.path.join(best_dir, "error_vs_epochs.png"),
    )

    # Plot fit and scatter
    for s_name, Xs, ys, preds in [("train", X_train, y_train, y_train_pred), ("val", X_val, y_val, y_val_pred), ("test", X_test, y_test, y_test_pred)]:
        if dim == 1:
            plot_regression_1d(Xs[:, 0], ys, preds, title=f"A1 {dataset_name} ({s_name})", filename=os.path.join(best_dir, f"fit_{s_name}.png"))
        else:
            plot_regression_2d(Xs, ys, preds, title=f"A1 {dataset_name} ({s_name})", filename=os.path.join(best_dir, f"fit_{s_name}.png"))
        plot_target_vs_model_scatter(ys, preds, title=f"A1 {dataset_name} Scatter ({s_name})", filename=os.path.join(best_dir, f"scatter_{s_name}.png"))

    print(f"Best A1 Config: {best_cfg['config_id']}")
    print(f"Val RMSE:  {best_val_metrics['rmse']:.4f} ({best_val_metrics['percent_rmse']:.2f}%)")
    print(f"Test RMSE: {best_test_metrics['rmse']:.4f} ({best_test_metrics['percent_rmse']:.2f}%) (Epochs: {best['epochs_run']})")

    return best


def main():
    for dataset_name, spec in DATASETS.items():
        run_dataset(dataset_name, spec["path"], spec["dim"])


if __name__ == "__main__":
    main()
