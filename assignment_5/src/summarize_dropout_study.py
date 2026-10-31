"""Summarize Dropout Study across Architectures for Caltech-101 Custom CNNs.

This script:
1. Reads architecture_comparison_metrics.json from custom_cnn/dropout*/ folders.
2. Writes a CSV of train/val/test metrics for each of the 9 (dropout, architecture) pairs.
3. Draws a grouped bar chart (x = architecture, grouped by dropout, 3 panels: train, val, test).
4. Determines the overall best custom CNN model using (val_acc, -val_loss, -param_count)
   and saves overall_best.json.
5. Reads vgg19/vgg19_metrics.json and saves q1_q2_test_comparison.png and .json,
   contrasting the best custom CNN against pretrained VGG19.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "partB"
CUSTOM_CNN_DIR = RESULTS_DIR / "custom_cnn"
VGG19_DIR = RESULTS_DIR / "vgg19"
CLASS_NAMES = ["butterfly", "car_side", "grand_piano", "scorpion", "starfish"]


def load_dropout_metrics(custom_cnn_dir: Path) -> list[dict]:
    """Load metrics from all available dropout subdirectories."""
    # Find all dropout folders
    dropout_dirs = sorted(
        [d for d in custom_cnn_dir.iterdir() if d.is_dir() and d.name.startswith("dropout")],
        key=lambda d: float(d.name.replace("dropout", "")) if d.name.replace("dropout", "").replace(".", "", 1).isdigit() else 0.0,
    )

    records = []
    for d_dir in dropout_dirs:
        metrics_file = d_dir / "architecture_comparison_metrics.json"
        summary_file = d_dir / "custom_cnn_architectures_summary.json"

        if not metrics_file.exists():
            continue

        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}

        dropout_str = d_dir.name.replace("dropout", "")
        dropout_val = float(dropout_str) if dropout_str else 0.0

        results = metrics.get("results", {})
        arch_summaries = summary.get("architectures", {})

        for arch_name in ("arch1", "arch2", "arch3"):
            if arch_name not in results:
                continue

            r = results[arch_name]
            s = arch_summaries.get(arch_name, {})

            # Retrieve validation loss from available fields
            val_loss = r.get("val_loss", r.get("best_val_loss", s.get("val_loss", s.get("best_val_loss", 0.0))))

            records.append({
                "folder": d_dir.name,
                "dropout": dropout_val,
                "architecture": arch_name,
                "train_acc": float(r.get("train_acc", 0.0)),
                "val_acc": float(r.get("val_acc", 0.0)),
                "test_acc": float(r.get("test_acc", 0.0)),
                "val_loss": float(val_loss),
                "param_count": int(r.get("param_count", 0)),
                "best_epoch": int(r.get("best_epoch", 0)),
                "total_epochs_run": int(r.get("total_epochs_run", 0)),
                "train_cm": r.get("train_cm", []),
                "val_cm": r.get("val_cm", []),
                "test_cm": r.get("test_cm", []),
            })

    return records


def write_csv_summary(records: list[dict], out_paths: list[Path]) -> None:
    """Write train/val/test metrics table to CSV."""
    fieldnames = [
        "dropout",
        "folder",
        "architecture",
        "train_acc",
        "val_acc",
        "test_acc",
        "val_loss",
        "param_count",
        "best_epoch",
        "total_epochs_run",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"Saved dropout study CSV to {out_path}")


def plot_grouped_bar_chart(records: list[dict], out_paths: list[Path]) -> None:
    """Draw a 3-panel grouped bar chart: train, val, test accuracy across architectures and dropouts."""
    dropouts = sorted(list({r["dropout"] for r in records}))
    architectures = ["arch1", "arch2", "arch3"]
    arch_labels = ["Arch 1\n(4 layers)", "Arch 2\n(5 layers)", "Arch 3\n(6 layers)"]

    # Colors for dropout values
    palette = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]
    dropout_colors = {d: palette[i % len(palette)] for i, d in enumerate(dropouts)}

    # Index by (dropout, arch)
    lookup = {(r["dropout"], r["architecture"]): r for r in records}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    splits = [
        ("train_acc", "Train Accuracy", axes[0]),
        ("val_acc", "Validation Accuracy", axes[1]),
        ("test_acc", "Test Accuracy", axes[2]),
    ]

    x = np.arange(len(architectures))
    n_dropouts = len(dropouts)
    total_width = 0.72
    bar_width = total_width / max(n_dropouts, 1)

    for metric_key, title, ax in splits:
        for i, d in enumerate(dropouts):
            offsets = x - (total_width / 2.0) + (i + 0.5) * bar_width
            vals = [lookup.get((d, a), {}).get(metric_key, 0.0) * 100 for a in architectures]
            bars = ax.bar(
                offsets,
                vals,
                width=bar_width * 0.92,
                label=f"Dropout {d:g}",
                color=dropout_colors[d],
                alpha=0.9,
                edgecolor="#1E293B",
            )
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        h + 1.2,
                        f"{h:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        fontweight="bold",
                    )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(arch_labels, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Architecture", fontsize=10)
        ax.set_ylim(0, 112)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].legend(title="Dropout Rate", loc="upper left", fontsize=8.5, title_fontsize=9)

    fig.suptitle(
        "Caltech-101 Custom CNN Dropout Study: Train, Val, and Test Accuracy by Architecture",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved grouped bar chart to {out_path}")
    plt.close(fig)


def select_overall_best(records: list[dict]) -> dict:
    """Select the overall best model using validation accuracy, negative validation loss, negative parameter count."""
    best = max(
        records,
        key=lambda r: (r["val_acc"], -r["val_loss"], -r["param_count"]),
    )
    return best


def save_vgg19_comparison(best_custom: dict, vgg_metrics_path: Path, out_fig_paths: list[Path], out_json_paths: list[Path]):
    """Generate side-by-side test confusion matrix comparison between best custom CNN and VGG19."""
    if not vgg_metrics_path.exists():
        print(f"Notice: {vgg_metrics_path} not found. Skipping VGG19 comparative plot.")
        return

    vgg_metrics = json.loads(vgg_metrics_path.read_text(encoding="utf-8"))
    vgg_test = vgg_metrics.get("test", {})
    vgg_acc = float(vgg_test.get("accuracy", 0.0))
    vgg_cm = np.asarray(vgg_test.get("confusion_matrix", np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))))

    custom_acc = float(best_custom["test_acc"])
    custom_cm = np.asarray(best_custom["test_cm"])

    comparison_json = {
        "custom_cnn": {
            "run_folder": best_custom["folder"],
            "dropout": best_custom["dropout"],
            "architecture": best_custom["architecture"],
            "optimizer": "Adam",
            "test_accuracy": custom_acc,
            "test_confusion_matrix": custom_cm.tolist(),
        },
        "vgg19": {
            "test_accuracy": vgg_acc,
            "test_confusion_matrix": vgg_cm.tolist(),
        },
    }

    for json_p in out_json_paths:
        json_p.parent.mkdir(parents=True, exist_ok=True)
        json_p.write_text(json.dumps(comparison_json, indent=2), encoding="utf-8")
        print(f"Saved comparative metrics JSON to {json_p}")

    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (
            axes[0],
            f"Best Custom CNN: {best_custom['architecture'].upper()} ({best_custom['folder']})\nTest Accuracy: {custom_acc * 100:.1f}%",
            custom_cm,
        ),
        (
            axes[1],
            f"Pretrained VGG19 Transfer Learning\nTest Accuracy: {vgg_acc * 100:.1f}%",
            vgg_cm,
        ),
    ]

    for ax, title, matrix in panels:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
        for r in range(num_classes):
            for c in range(num_classes):
                ax.text(
                    c,
                    r,
                    str(int(matrix[r, c])),
                    ha="center",
                    va="center",
                    color="white" if matrix[r, c] > thresh else "black",
                    fontweight="bold",
                )

    fig.suptitle(
        "Comparative Test Set Performance: Best Custom CNN vs. Pretrained VGG19",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    for fig_p in out_fig_paths:
        fig_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_p, dpi=180, bbox_inches="tight")
        print(f"Saved comparative test confusion matrix to {fig_p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize Caltech-101 Custom CNN Dropout Study")
    parser.add_argument("--custom_dir", type=Path, default=CUSTOM_CNN_DIR, help="Directory containing dropout* runs")
    args = parser.parse_args()

    records = load_dropout_metrics(args.custom_dir)
    if not records:
        print(f"Error: No architecture comparison metrics found under {args.custom_dir}")
        return

    print(f"Loaded {len(records)} architecture-dropout evaluations.")

    # 1. Write CSV summary
    csv_paths = [
        RESULTS_DIR / "dropout_study_summary.csv",
        args.custom_dir / "dropout_study_summary.csv",
    ]
    write_csv_summary(records, csv_paths)

    # 2. Draw grouped bar chart
    chart_paths = [
        RESULTS_DIR / "dropout_study_grouped_bar_chart.png",
        args.custom_dir / "dropout_study_grouped_bar_chart.png",
    ]
    plot_grouped_bar_chart(records, chart_paths)

    # 3. Select overall best model using selection rule
    best_candidate = select_overall_best(records)
    print("\n" + "=" * 60)
    print(f"OVERALL BEST MODEL: {best_candidate['architecture'].upper()} in {best_candidate['folder']}")
    print(f"Val Acc: {best_candidate['val_acc'] * 100:.1f}% | Val Loss: {best_candidate['val_loss']:.4f} | Test Acc: {best_candidate['test_acc'] * 100:.1f}%")
    print(f"Parameters: {best_candidate['param_count']:,} | Best Epoch: {best_candidate['best_epoch']}")
    print("=" * 60 + "\n")

    overall_best_data = {
        "folder": best_candidate["folder"],
        "custom_run": best_candidate["folder"],
        "architecture": best_candidate["architecture"],
        "dropout": best_candidate["dropout"],
        "train_acc": best_candidate["train_acc"],
        "val_acc": best_candidate["val_acc"],
        "val_loss": best_candidate["val_loss"],
        "test_acc": best_candidate["test_acc"],
        "param_count": best_candidate["param_count"],
        "best_epoch": best_candidate["best_epoch"],
        "total_epochs_run": best_candidate["total_epochs_run"],
        "test_cm": best_candidate["test_cm"],
    }

    best_json_paths = [
        RESULTS_DIR / "overall_best.json",
        args.custom_dir / "overall_best.json",
    ]
    for p in best_json_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(overall_best_data, indent=2), encoding="utf-8")
        print(f"Saved overall best model info to {p}")

    # 4. Save VGG19 comparison figure and json
    vgg_metrics_path = VGG19_DIR / "vgg19_metrics.json"
    comp_fig_paths = [
        VGG19_DIR / "q1_q2_test_comparison.png",
        VGG19_DIR / "custom_cnn_vs_vgg19_test_comparison.png",
        RESULTS_DIR / "q1_q2_test_comparison.png",
    ]
    comp_json_paths = [
        VGG19_DIR / "q1_q2_test_comparison.json",
        VGG19_DIR / "custom_cnn_vs_vgg19_test_comparison.json",
        RESULTS_DIR / "q1_q2_test_comparison.json",
    ]
    save_vgg19_comparison(best_candidate, vgg_metrics_path, comp_fig_paths, comp_json_paths)


if __name__ == "__main__":
    main()

