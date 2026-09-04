from fcnn import FCNN
from utils.metrics import classification_metrics


def architecture_grid(input_dim, output_dim, hidden_layer_size_options,
                       hidden_activation="logistic", output_activation="logistic",
                       learning_rates=(0.05,), epochs_list=(500,),
                       stopping_threshold=None):
    # Generates configuration dictionaries for model sweeping
    configs = []
    for hidden_sizes in hidden_layer_size_options:
        hidden_str = "-".join(str(s) for s in hidden_sizes)
        for lr in learning_rates:
            for ep in epochs_list:
                layer_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
                cfg_id = f"H{hidden_str}_{hidden_activation}_LR{lr}_EP{ep}"
                configs.append({
                    "layer_sizes": layer_sizes,
                    "hidden_activation": hidden_activation,
                    "output_activation": output_activation,
                    "learning_rate": lr,
                    "epochs": ep,
                    "stopping_threshold": stopping_threshold,
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
            stopping_threshold=cfg["stopping_threshold"],
            seed=42
        )
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_metrics = classification_metrics(y_val, y_val_pred, num_classes)

        sweep_results.append({
            "config": cfg,
            "model": model,
            "val_metrics": val_metrics,
            "epochs_run": len(model.errors)
        })

    return sweep_results


def select_best(sweep_results, metric_key="overall_accuracy"):
    # Selects the best performing model based on the chosen validation metric
    return max(sweep_results, key=lambda entry: entry["val_metrics"][metric_key])
