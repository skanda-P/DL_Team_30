from fcnn import FCNN
from utils.metrics import classification_metrics, rmse, percent_rmse


def architecture_grid(input_dim, output_dim, hidden_layer_size_options,
                       hidden_activation="logistic", output_activation="logistic",
                       learning_rates=(0.01, 0.05, 0.1), max_epochs=2000,
                       stopping_threshold=0.0001, patience=5):
    # Generates configuration dictionaries for model sweeping
    configs = []
    for hidden_sizes in hidden_layer_size_options:
        hidden_str = "-".join(str(s) for s in hidden_sizes)
        for lr in learning_rates:
            layer_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
            cfg_id = f"H{hidden_str}_{hidden_activation}_LR{lr}"
            configs.append({
                "layer_sizes": layer_sizes,
                "hidden_activation": hidden_activation,
                "output_activation": output_activation,
                "learning_rate": lr,
                "epochs": max_epochs,
                "stopping_threshold": stopping_threshold,
                "patience": patience,
                "config_id": cfg_id
            })
    return configs


def run_sweep(X_train, y_train, X_val, y_val, architectures, num_classes):
    # Trains each architecture on train split and evaluates on validation split
    sweep_results = []
    for cfg in architectures:
        model = FCNN(
            layer_sizes=cfg["layer_sizes"],
            hidden_activation=cfg["hidden_activation"],
            output_activation=cfg["output_activation"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            stopping_threshold=cfg.get("stopping_threshold"),
            patience=cfg.get("patience", 5),
            seed=42
        )
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_metrics = classification_metrics(y_val, y_val_pred, num_classes)

        sweep_results.append({
            "config": cfg,
            "model": model,
            "val_metrics": val_metrics,
            "epochs_run": model.epochs_run
        })

    return sweep_results


def select_best(sweep_results, metric_key="overall_accuracy"):
    # Selects the best performing model based on the chosen validation metric
    return max(sweep_results, key=lambda entry: entry["val_metrics"][metric_key])

def run_regression_sweep(X_train, y_train, X_val, y_val, architectures):
    # Trains each regression architecture on the train split and evaluates
    # it on the validation split using RMSE / %RMSE.
    sweep_results = []
    for cfg in architectures:
        model = FCNN(
            layer_sizes=cfg["layer_sizes"],
            hidden_activation=cfg["hidden_activation"],
            output_activation=cfg["output_activation"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            stopping_threshold=cfg.get("stopping_threshold"),
            patience=cfg.get("patience", 5),
            seed=42
        )
        model.fit(X_train, y_train)

        # The assignment asks for train AND validation RMSE/%RMSE for every
        # architecture in the sweep (not just the winning one), so both are
        # computed and returned here.
        y_train_pred = model.predict(X_train)
        train_metrics = {
            "rmse": rmse(y_train, y_train_pred),
            "percent_rmse": percent_rmse(y_train, y_train_pred),
        }

        y_val_pred = model.predict(X_val)
        val_metrics = {
            "rmse": rmse(y_val, y_val_pred),
            "percent_rmse": percent_rmse(y_val, y_val_pred),
        }

        sweep_results.append({
            "config": cfg,
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "epochs_run": model.epochs_run
        })

    return sweep_results 


def select_best_regression(sweep_results, metric_key="rmse"):
    # Lower RMSE / %RMSE is better, unlike the classification metrics above.
    return min(sweep_results, key=lambda entry: entry["val_metrics"][metric_key])
