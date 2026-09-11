import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import numpy as np

# Ensure src/ directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DEFAULT_RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

from data_loader import get_data_tensors
from models import build_model, ARCHITECTURES, ARCH_GROUPS, ACTIVATION_CHOICES
from utils.metrics import classification_metrics, print_classification_report
from utils.plotting import plot_error_vs_epochs
from utils.training_config import MAX_EPOCHS, STOPPING_THRESHOLD, PATIENCE


OPTIMIZERS = {
    "sgd": {
        "display_name": "SGD (batch_size=1)",
        "batch_size": 1,
        "builder": lambda params: torch.optim.SGD(params, lr=0.001)
    },
    "bgd": {
        "display_name": "Batch GD (batch_size=N)",
        "batch_size": "total",
        "builder": lambda params: torch.optim.SGD(params, lr=0.001)
    },
    "momentum": {
        "display_name": "SGD + Momentum (batch_size=1)",
        "batch_size": 1,
        "builder": lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=False)
    },
    "nag": {
        "display_name": "SGD + NAG (batch_size=1)",
        "batch_size": 1,
        "builder": lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    },
    "adagrad": {
        "display_name": "AdaGrad (batch_size=N)",
        "batch_size": "total",
        "builder": lambda params: torch.optim.Adagrad(params, lr=0.001)
    },
    "rmsprop": {
        "display_name": "RMSProp (batch_size=N)",
        "batch_size": "total",
        "builder": lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.99, eps=1e-8)
    },
    "adam": {
        "display_name": "Adam (batch_size=1)",
        "batch_size": 1,
        "builder": lambda params: torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    }
}


def evaluate(model, X, y, criterion, batch_size=1024):
    """
    Evaluates loss, accuracy, and predicted labels for a dataset split.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    num_samples = len(X)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)

    avg_loss = total_loss / num_samples
    all_preds = torch.cat(all_preds).cpu().numpy()
    y_np = y.cpu().numpy()
    accuracy = (all_preds == y_np).mean()

    return avg_loss, accuracy, all_preds


def train_single_run(arch_name, optimizer_key, activation="relu", data_dir=None, results_dir=None,
                     stopping_threshold=STOPPING_THRESHOLD, max_epochs=MAX_EPOCHS,
                     patience=1, device=None, verbose=True, seed=42):
    """
    Trains an FCNN architecture using a specified activation and optimizer.
    Guarantees:
      - Uses identical initial random weights across all optimizers for this (architecture, activation).
      - Stops when |loss_t - loss_{t-1}| < stopping_threshold (1e-4).
      - Configures exact hyperparameters specified in the assignment PDF.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    if device is None:
        device = "cpu"

    if optimizer_key not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{optimizer_key}'. Available: {list(OPTIMIZERS.keys())}")

    opt_info = OPTIMIZERS[optimizer_key]
    display_name = opt_info["display_name"]
    batch_mode = opt_info["batch_size"]

    max_epochs_str = str(max_epochs) if max_epochs is not None else "Unlimited"
    if verbose:
        print(f"\n=======================================================")
        print(f"Architecture: {arch_name} | Activation: {activation} | Optimizer: {display_name}")
        print(f"Device: {device} | Threshold: {stopping_threshold} | Max Epochs: {max_epochs_str}")
        print(f"=======================================================")

    # Load dataset tensors on device
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx, idx_to_class = get_data_tensors(
        data_dir=data_dir, device=device
    )
    N_train = len(X_train)

    # Build model and load identical initial weights for this (architecture, activation)
    model = build_model(arch_name, num_classes=len(class_to_idx), activation=activation, seed=seed).to(device)


    # Build optimizer
    optimizer = opt_info["builder"](model.parameters())
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    consecutive_stops = 0
    converged = False
    start_time = time.time()

    # Training loop
    epoch = 0
    while True:
        epoch += 1
        if max_epochs is not None and epoch > max_epochs:
            break

        model.train()

        if batch_mode == "total":
            # Batch Gradient Descent / AdaGrad / RMSProp (batch_size = N)
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            avg_loss = loss.item()
        else:
            # Stochastic Gradient Descent (batch_size = 1)
            perm = torch.randperm(N_train, device=device)
            total_loss = 0.0
            for idx in perm:
                optimizer.zero_grad()
                out = model(X_train[idx : idx + 1])
                sample_loss = criterion(out, y_train[idx : idx + 1])
                sample_loss.backward()
                optimizer.step()
                total_loss += sample_loss.item()
            avg_loss = total_loss / N_train

        epoch_losses.append(avg_loss)

        # Evaluate stopping criterion: |avg_loss_{epoch} - avg_loss_{epoch-1}| < threshold
        if epoch > 1:
            diff = abs(epoch_losses[-1] - epoch_losses[-2])
            if diff < stopping_threshold:
                consecutive_stops += 1
                if consecutive_stops >= patience:
                    converged = True
            else:
                consecutive_stops = 0
        else:
            diff = float("inf")

        if verbose and (epoch % 5 == 0 or epoch <= 5 or converged or (max_epochs is not None and epoch == max_epochs)):
            diff_str = f"{diff:.6f}" if diff != float("inf") else "N/A"
            print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f} | |Diff|: {diff_str} "
                  f"| Below Threshold Count: {consecutive_stops}/{patience}")

        if converged:
            if verbose:
                print(f"[*] Convergence reached at epoch {epoch}! (|Diff| = {diff:.6f} < {stopping_threshold})")
            break

    elapsed_time = time.time() - start_time
    epochs_run = len(epoch_losses)

    # Final evaluations
    train_loss, train_acc, train_preds = evaluate(model, X_train, y_train, criterion)
    val_loss, val_acc, val_preds = evaluate(model, X_val, y_val, criterion)

    num_classes = len(class_to_idx)
    val_metrics = classification_metrics(y_val.cpu().numpy(), val_preds, num_classes)
    train_metrics = classification_metrics(y_train.cpu().numpy(), train_preds, num_classes)

    if verbose:
        print(f"\nCompleted in {elapsed_time:.2f}s across {epochs_run} epochs.")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")

    # Save artifacts
    save_dir = os.path.join(results_dir, arch_name, activation, optimizer_key)
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save loss history in text format
    loss_lines = [f"Epoch {ep:4d}: Loss = {l:.6f}" for ep, l in enumerate(epoch_losses, 1)]
    with open(os.path.join(save_dir, "loss_history.txt"), "w") as f:
        f.write(f"Architecture:          {arch_name}\n")
        f.write(f"Activation Function:   {activation}\n")
        f.write(f"Optimizer:             {display_name}\n")
        f.write(f"Epochs to Converge:    {epochs_run}\n")
        f.write(f"Stopped by Threshold:  {'Yes' if converged else 'No (max epochs)'}\n")
        f.write(f"Elapsed Time (s):      {elapsed_time:.2f}\n")
        f.write("\nEpoch Training Losses:\n")
        f.write("----------------------\n")
        f.write("\n".join(loss_lines) + "\n")

    # Save evaluation metrics in text format
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write("======================================================================\n")
        f.write("                       MODEL EVALUATION METRICS                       \n")
        f.write("======================================================================\n\n")
        f.write(f"Architecture:              {arch_name}\n")
        f.write(f"Activation Function:       {activation}\n")
        f.write(f"Optimizer:                 {display_name}\n")
        f.write(f"Epochs to Converge:        {epochs_run}\n")
        f.write(f"Stopped by Threshold:      {'Yes' if converged else 'No (max epochs)'}\n")
        f.write(f"Elapsed Time:              {elapsed_time:.2f} seconds\n\n")
        f.write("--- Performance Summary ---\n")
        f.write(f"Training Loss:             {train_loss:.6f}\n")
        f.write(f"Training Accuracy:         {train_acc * 100:.2f}%\n")
        f.write(f"Validation Loss:           {val_loss:.6f}\n")
        f.write(f"Validation Accuracy:       {val_acc * 100:.2f}%\n")
        f.write(f"Validation Macro F1:       {val_metrics['macro_f_measure'] * 100:.2f}%\n")
        f.write(f"Validation Micro F1:       {val_metrics['micro_f_measure'] * 100:.2f}%\n\n")
        f.write("--- Per-Class Metrics (Validation Split) ---\n")
        for c_idx, c_name in idx_to_class.items():
            p_val = val_metrics["class_precision"][c_idx]
            r_val = val_metrics["class_recall"][c_idx]
            f_val = val_metrics["class_f_measure"][c_idx]
            f.write(f"Digit '{c_name}': Precision = {p_val:.4f}, Recall = {r_val:.4f}, F1 = {f_val:.4f}\n")
        f.write("======================================================================\n")

    # Plot single error vs epochs
    plot_error_vs_epochs(
        epoch_losses,
        title=f"Average Error vs Epochs: {arch_name} ({activation}) - {display_name}",
        filename=os.path.join(save_dir, "error_vs_epochs.png")
    )

    return {
        "arch": arch_name,
        "activation": activation,
        "optimizer": optimizer_key,
        "display_name": display_name,
        "epochs_run": epochs_run,
        "converged": converged,
        "elapsed_time": elapsed_time,
        "losses": epoch_losses,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
        "model": model,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }


def main():
    parser = argparse.ArgumentParser(description="Train FCNN with various optimizers and activations.")
    parser.add_argument("--arch", type=str, nargs="+", default=None,
                        help="Architecture(s) to train: 'all', '3_layers', '4_layers', '5_layers', or specific arch keys (default: 'all')")
    parser.add_argument("--activation", type=str, nargs="+", default=None,
                        help="Activation function(s) or 'all' (choices: relu, tanh, sigmoid; default: 'all')")
    parser.add_argument("--optimizer", type=str, nargs="+", default=None,
                        help="Optimizer(s) to use or 'all' (choices: sgd, bgd, momentum, nag, adagrad, rmsprop, adam; default: 'all')")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=STOPPING_THRESHOLD,
                        help="Stopping threshold |L_t - L_{t-1}| < threshold")
    def parse_max_epochs(v):
        if v is None or str(v).lower() in ("none", "inf", "unlimited", "0", "-1"):
            return None
        return int(v)

    parser.add_argument("--max_epochs", type=parse_max_epochs, default=MAX_EPOCHS,
                        help="Maximum epochs per run (default: 10000, or 'none' for unlimited)")
    parser.add_argument("--patience", type=int, default=1, help="Consecutive epochs below threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight initialization")

    args = parser.parse_args()

    if args.arch is None or "all" in args.arch:
        archs_to_run = ARCH_GROUPS["all"]
    else:
        archs_to_run = []
        for a in args.arch:
            if a in ARCH_GROUPS:
                archs_to_run.extend(ARCH_GROUPS[a])
            else:
                archs_to_run.append(a)

    opts_to_run = list(OPTIMIZERS.keys()) if (args.optimizer is None or "all" in args.optimizer) else args.optimizer
    acts_to_run = list(ACTIVATION_CHOICES) if (args.activation is None or "all" in args.activation) else args.activation

    # If running multiple architectures or multiple optimizers, run full comparative suite
    if len(archs_to_run) > 1 or len(opts_to_run) > 1 or len(acts_to_run) > 1:
        from compare_optimizers import run_experiments
        run_experiments(
            architectures=archs_to_run,
            activations=acts_to_run,
            optimizers=opts_to_run,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            stopping_threshold=args.threshold,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
            seed=args.seed
        )
    else:
        # Single individual model run
        train_single_run(
            arch_name=archs_to_run[0],
            optimizer_key=opts_to_run[0],
            activation=acts_to_run[0],
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            stopping_threshold=args.threshold,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
