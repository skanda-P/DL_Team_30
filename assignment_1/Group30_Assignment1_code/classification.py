import os
import numpy as np

from multiclass import OneVsOneClassifier
from utils.data_utils import load_LS_data, load_nls_data, stratified_train_test_split
from utils.metrics import classification_metrics, print_classification_report
from utils.plotting import plot_decision_regions, plot_error_vs_epochs


DATA_DIR = "data"
LS_DIR = os.path.join(DATA_DIR, "Classification", "LS_Group30")
NLS_FILE = os.path.join(DATA_DIR, "Classification", "NLS_Group30.txt")

ACTIVATIONS = ["logistic", "tanh"]
LEARNING_RATES = [0.001, 0.01, 0.1, 0.2]
EPOCHS_LIST = [500, 1000, 1500, 2000]
TEST_RATIO = 0.3
SEED = 42

def run_dataset(dataset_name, X, y, activation, lr, epochs):

    out_dir = os.path.abspath(os.path.join("results", dataset_name, f"{activation}_LR{lr}_EP{epochs}"))
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_ratio=TEST_RATIO, seed=SEED
    )
    all_classes = np.unique(y)
    num_classes = len(all_classes)


    clf = OneVsOneClassifier(activation=activation, learning_rate=lr, epochs=epochs)
    clf.fit(X_train, y_train)


    for (class_a, class_b), model in clf.classifiers.items():
        plot_error_vs_epochs(
            model.errors,
            title=f"{dataset_name} ({activation}): Error (Class {class_a} vs {class_b})",
            filename=os.path.join(out_dir, f"error_class{class_a}_vs_class{class_b}.png"),
        )


    for (class_a, class_b) in clf.classifiers:
        mask = (y_train == class_a) | (y_train == class_b)

        def predict_fn(Xp, _a=class_a, _b=class_b):
            return clf.predict_pair_labels(_a, _b, Xp)

        plot_decision_regions(
            X_train[mask], y_train[mask], predict_fn,
            title=f"{dataset_name} ({activation}): Decision (Class {class_a} vs {class_b})",
            filename=os.path.join(out_dir, f"decision_class{class_a}_vs_class{class_b}.png"),
            all_classes=all_classes,
        )


    plot_decision_regions(
        X_train, y_train, clf.predict,
        title=f"{dataset_name} ({activation}): Combined Decision",
        filename=os.path.join(out_dir, "decision_combined.png"),
        all_classes=all_classes,
    )


    y_pred_test = clf.predict(X_test)
    metrics = classification_metrics(y_test, y_pred_test, num_classes)

    with open(os.path.join(out_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Configuration: {dataset_name} | {activation} | LR: {lr} | EP: {epochs}\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    return clf, metrics

def main():
    X_ls, y_ls = load_LS_data(LS_DIR)
    X_nls, y_nls = load_nls_data(NLS_FILE)
    datasets = {"LS": (X_ls, y_ls), "NLS": (X_nls, y_nls)}


    results = {}
    for dataset_name, (X, y) in datasets.items():
        for activation in ACTIVATIONS:
            for lr in LEARNING_RATES:
                for epochs in EPOCHS_LIST:
                    results[(dataset_name, activation, lr, epochs)] = run_dataset(
                        dataset_name, X, y, activation, lr, epochs
                    )
    return results

if __name__ == "__main__":
    main()
