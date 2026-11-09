import os
import re
import argparse
import matplotlib.pyplot as plt

OPTIMIZER_CONFIGS = {
    "momentum": {"label": "SGD + Momentum (batch_size=1)", "color": "#ff7f0e", "marker": "^"},
    "nag":      {"label": "SGD + NAG (batch_size=1)",      "color": "#d62728", "marker": "v"},
    "adam":     {"label": "Adam (batch_size=1)",          "color": "#8c564b", "marker": "*"},
    "sgd":      {"label": "SGD (batch_size=1)",           "color": "#1f77b4", "marker": "o"},
    "rmsprop":  {"label": "RMSProp (batch_size=N)",       "color": "#9467bd", "marker": "P"},
    "adagrad":  {"label": "AdaGrad (batch_size=N)",       "color": "#2ca02c", "marker": "D"},
    "bgd":      {"label": "Batch GD (batch_size=N)",      "color": "#000000", "marker": "s"}
}

ORDER_DISPLAY = ["momentum", "nag", "adam", "sgd", "rmsprop", "adagrad", "bgd"]


def parse_loss_history(filepath):
    if not os.path.exists(filepath):
        return None
    meta = {}
    losses = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line and not line.startswith("Epoch"):
                parts = line.split(":", 1)
                meta[parts[0].strip()] = parts[1].strip()
            elif line.startswith("Epoch"):
                m = re.search(r"Epoch\s+(\d+):\s*Loss\s*=\s*([\d\.]+)", line)
                if m:
                    losses.append(float(m.group(2)))
    init_loss = float(meta.get("Initial Loss (pre-training)", losses[0] if losses else 0.0))
    return {"init_loss": init_loss, "losses": losses, "epochs": len(losses)}


def load_architecture_data(results_dir, arch_name, act_name="tanh"):
    arch_dir = os.path.join(results_dir, arch_name, act_name)
    if not os.path.isdir(arch_dir):
        return None

    data = {}
    for opt_key in ORDER_DISPLAY:
        opt_path = os.path.join(arch_dir, opt_key)
        lh_file = os.path.join(opt_path, "loss_history.txt")
        lh_data = parse_loss_history(lh_file)
        if lh_data is None or len(lh_data["losses"]) == 0:
            continue

        cfg = OPTIMIZER_CONFIGS.get(opt_key, {
            "label": opt_key,
            "color": "#333333",
            "marker": "o"
        })

        data[opt_key] = {
            "label": cfg["label"],
            "color": cfg["color"],
            "marker": cfg["marker"],
            "init_loss": lh_data["init_loss"],
            "losses": lh_data["losses"],
            "epochs": lh_data["epochs"]
        }
    return data


def plot_epochs_0_to_100(data, arch_name, act_name, output_path, dpi=300):
    plt.figure(figsize=(11, 6.5), dpi=dpi)

    for opt_key, item in data.items():
        curve_losses = [item["init_loss"]] + item["losses"]
        epochs = list(range(0, item["epochs"] + 1))

        clipped_epochs = [e for e in epochs if e <= 100]
        clipped_losses = curve_losses[:len(clipped_epochs)]

        plt.plot(
            clipped_epochs,
            clipped_losses,
            label=f"{item['label']} ({item['epochs']} ep)",
            color=item["color"],
            linestyle="-",
            linewidth=1.0,
            alpha=0.9
        )

        end_epoch = item["epochs"]
        if end_epoch <= 100:
            end_loss = item["losses"][-1]
            plt.scatter([end_epoch], [end_loss], color=item["color"], s=45, marker=item["marker"], zorder=5)
            plt.axvline(x=end_epoch, color=item["color"], linestyle="--", linewidth=0.8, alpha=0.55, zorder=2)

    first_item = next(iter(data.values()))
    plt.scatter([0], [first_item["init_loss"]], color="#555555", s=50, marker="X", zorder=6, label=f"Initial Loss ({first_item['init_loss']:.4f})")

    plt.xlim(-2, 102)
    plt.xlabel("Epoch Number", fontsize=12, fontweight="bold")
    plt.ylabel("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    plt.title(f"Average Training Error vs. Epochs (Epochs 0 to 100): {arch_name.upper()} ({act_name.upper()})", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_logscale_full_history(data, arch_name, act_name, output_path, dpi=300):
    plt.figure(figsize=(11, 6.5), dpi=dpi)

    for opt_key, item in data.items():
        curve_losses = [item["init_loss"]] + item["losses"]
        epochs = list(range(0, item["epochs"] + 1))

        plt.plot(
            epochs,
            curve_losses,
            label=f"{item['label']} ({item['epochs']} ep)",
            color=item["color"],
            linestyle="-",
            linewidth=1.0,
            alpha=0.9
        )

        end_epoch = item["epochs"]
        end_loss = item["losses"][-1]
        plt.scatter([end_epoch], [end_loss], color=item["color"], s=45, marker=item["marker"], zorder=5)
        plt.axvline(x=end_epoch, color=item["color"], linestyle="--", linewidth=0.8, alpha=0.55, zorder=2)

    first_item = next(iter(data.values()))
    plt.scatter([0], [first_item["init_loss"]], color="#555555", s=50, marker="X", zorder=6, label=f"Initial Loss ({first_item['init_loss']:.4f})")

    plt.yscale("log")
    plt.xlabel("Epoch Number", fontsize=12, fontweight="bold")
    plt.ylabel("Cross-Entropy Loss (Log Scale)", fontsize=12, fontweight="bold")
    plt.title(f"Average Training Error vs. Epochs (Full Horizon, Log Scale): {arch_name.upper()} ({act_name.upper()})", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate superimposed error plots (0-100 epochs and full logscale).")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--arch", type=str, default="all")
    parser.add_argument("--act", type=str, default="tanh")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.exists(results_dir):
        print(f"Results directory not found at {results_dir}")
        return

    if args.arch == "all":
        arch_names = sorted([
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("arch_")
        ])
    else:
        arch_names = [args.arch]

    for arch_name in arch_names:
        data = load_architecture_data(results_dir, arch_name, args.act)
        if not data:
            continue

        plot_0_to_100_path = os.path.join(results_dir, f"{arch_name}_{args.act}_error_vs_epochs_0_to_100.png")
        plot_epochs_0_to_100(data, arch_name, args.act, plot_0_to_100_path, dpi=args.dpi)

        plot_logscale_path = os.path.join(results_dir, f"{arch_name}_{args.act}_error_vs_epochs_logscale_full.png")
        plot_logscale_full_history(data, arch_name, args.act, plot_logscale_path, dpi=args.dpi)

        print(f"Generated plots for {arch_name} ({args.act})")


if __name__ == "__main__":
    main()
