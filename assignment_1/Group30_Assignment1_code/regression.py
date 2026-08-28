import os
from perceptron import Perceptron
from utils.data_utils import load_regression_csv, train_test_split
from utils.metrics import percent_rmse, rmse
from utils.plotting import (
    plot_error_vs_epochs, plot_regression_1d, plot_regression_2d, plot_target_vs_model_scatter
)


DATA_DIR = "data"
DATASETS = {
    "Univariate": {"path": os.path.join(DATA_DIR, "Regression", "UnivariateData", "30.csv"), "dim": 1},
    "Bivariate": {"path": os.path.join(DATA_DIR, "Regression", "BivariateData", "30.csv"), "dim": 2},
}

LEARNING_RATES = [0.01, 0.05]
EPOCHS_LIST = [500, 1000]
TEST_RATIO = 0.3
SEED = 42

def report_rmse(y_true, y_pred):
    return rmse(y_true, y_pred), percent_rmse(y_true, y_pred)

def run_dataset(dataset_name, path, dim, lr, epochs):

    out_dir = os.path.abspath(os.path.join("results", dataset_name, f"linear_LR{lr}_EP{epochs}"))
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_regression_csv(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=TEST_RATIO, seed=SEED)


    model = Perceptron(learning_rate=lr, epochs=epochs, activation="linear")
    model.fit(X_train, y_train)


    plot_error_vs_epochs(
        model.errors,
        title=f"{dataset_name}: Average Error vs Epochs",
        filename=os.path.join(out_dir, "error_vs_epoch.png"),
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    train_rmse, train_prmse = report_rmse(y_train, y_train_pred)
    test_rmse, test_prmse = report_rmse(y_test, y_test_pred)

    with open(os.path.join(out_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Configuration: {dataset_name} | Linear | LR: {lr} | EP: {epochs}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train RMSE: {train_rmse:.4f} | Train %RMSE: {train_prmse:.2f}%\n")
        f.write(f"Test RMSE: {test_rmse:.4f} | Test %RMSE: {test_prmse:.2f}%\n")


    if dim == 1:
        plot_regression_1d(X_train[:, 0], y_train, y_train_pred, title=f"{dataset_name} Train", filename=os.path.join(out_dir, "target_vs_model_train.png"))
        plot_regression_1d(X_test[:, 0], y_test, y_test_pred, title=f"{dataset_name} Test", filename=os.path.join(out_dir, "target_vs_model_test.png"))
    else:
        plot_regression_2d(X_train, y_train, y_train_pred, title=f"{dataset_name} Train", filename=os.path.join(out_dir, "target_vs_model_train.png"))
        plot_regression_2d(X_test, y_test, y_test_pred, title=f"{dataset_name} Test", filename=os.path.join(out_dir, "target_vs_model_test.png"))


    plot_target_vs_model_scatter(y_train, y_train_pred, title=f"{dataset_name} Scatter Train", filename=os.path.join(out_dir, "scatter_train.png"))
    plot_target_vs_model_scatter(y_test, y_test_pred, title=f"{dataset_name} Scatter Test", filename=os.path.join(out_dir, "scatter_test.png"))

    return model

def main():
    results = {}
    for dataset_name, spec in DATASETS.items():
        for lr in LEARNING_RATES:
            for epochs in EPOCHS_LIST:
                results[(dataset_name, lr, epochs)] = run_dataset(
                    dataset_name, spec["path"], spec["dim"], lr, epochs
                )
    return results

if __name__ == "__main__":
    main()
