from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import (
    discover_dataset,
    export_all_handwritten_normalized_images,
    normalize_handwritten_xy,
    parse_cv_file,
    parse_handwritten_file,
    render_stroke_on_axis,
)
from utils.io import load_json, save_csv_and_md

INVERT_Y = False


def load_curve(run_dir: str) -> Tuple[np.ndarray, np.ndarray, str]:
    p = Path(run_dir)
    m_path, h_path = p / "metrics.json", p / "history.json"
    if not (m_path.is_file() and h_path.is_file()):
        return np.array([]), np.array([]), "missing"
    met = load_json(m_path)
    hist = load_json(h_path)
    pairs = [(h["epoch"], h["train_error"]) for h in hist if np.isfinite(h["train_error"])]
    if not pairs:
        return np.array([]), np.array([]), met.get("stop_reason", "unknown")
    ep, err = zip(*pairs)
    return np.array(ep, dtype=np.int64), np.array(err, dtype=np.float64), met.get("stop_reason", "unknown")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
    normalized: bool = False,
) -> None:
    cm_arr = np.asarray(cm, dtype=np.float64)
    n_cls = len(class_names)
    if normalized:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        disp = np.divide(cm_arr, np.where(row_sums == 0, 1.0, row_sums))
    else:
        disp = cm_arr

    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=220)
    im = ax.imshow(disp, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if normalized else None)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n_cls),
        yticks=np.arange(n_cls),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = disp.max() / 2.0 if disp.max() > 0 else 0.5
    for i in range(n_cls):
        for j in range(n_cls):
            txt = f"{disp[i, j] * 100:.1f}%\n({int(cm_arr[i, j])})" if normalized else f"{int(cm_arr[i, j])}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8.5 if normalized else 10,
                fontweight="bold",
                color="white" if disp[i, j] > thresh else "black",
            )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_error_curve(run_dir: str, title: str, out_path: Path) -> None:
    ep, err, reason = load_curve(run_dir)
    if len(ep) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(ep, err, color="#1f77b4", linewidth=2.0, label=reason)
    if reason == "max_epochs_cap":
        ax.scatter([ep[-1]], [err[-1]], marker="^", s=45, color="darkorange", zorder=5)
    elif reason == "diverged":
        ax.scatter([ep[-1]], [err[-1]], marker="x", s=55, color="red", zorder=5)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Training Error (Cross-Entropy)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    pos = err[err > 0]
    if len(pos) > 0 and (pos.max() / max(float(pos.min()), 1e-12)) > 15.0:
        ax.set_yscale("log")
        ax.set_ylabel("Average Training Error (Log Scale)")
        fig.savefig(out_path.with_name(f"{out_path.stem}_log{out_path.suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage1_overview(summaries: List[Dict[str, Any]], title: str, out_path: Path) -> None:
    s1 = [s for s in summaries if s["stage"] == "stage1_arch"]
    if not s1:
        return
    layers = sorted({s["num_layers"] for s in s1})
    hiddens = sorted({s["hidden_size"] for s in s1})
    colors = {h: plt.cm.viridis(i / max(len(hiddens) - 1, 1)) for i, h in enumerate(hiddens)}

    fig, axes = plt.subplots(1, len(layers), figsize=(5.0 * len(layers), 4.0), dpi=220, sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, nl in zip(axes, layers):
        for s in sorted([x for x in s1 if x["num_layers"] == nl], key=lambda x: x["hidden_size"]):
            ep, err, reason = load_curve(s["run_dir"])
            if len(ep) == 0:
                continue
            lbl = f"hidden={s['hidden_size']}" + ("*" if reason != "converged" else "")
            ax.plot(ep, err, color=colors[s["hidden_size"]], linewidth=1.8, label=lbl)
            if reason != "converged":
                ax.scatter([ep[-1]], [err[-1]], marker="^", s=36, color=colors[s["hidden_size"]])
        ax.set_title(f"num_layers = {nl}", fontsize=10.5)
        ax.set_xlabel("Epochs")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Average Training Error")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage2_overview(summaries: List[Dict[str, Any]], title: str, out_path: Path) -> None:
    s2 = [s for s in summaries if s["stage"] == "stage2_opt"]
    if not s2:
        return
    s1 = [s for s in summaries if s["stage"] == "stage1_arch" and not s.get("diverged")]
    ref = max(s1, key=lambda s: (s["val_acc"], -s["val_loss"], -s["n_params"])) if s1 else None

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=220)
    if ref is not None:
        ep_r, err_r, _ = load_curve(ref["run_dir"])
        if len(ep_r) > 0:
            ax.plot(ep_r, err_r, color="black", linestyle="--", linewidth=2.0, label="Stage 1 Best (Ref)")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(s2), 1)))
    for i, s in enumerate(s2):
        ep, err, reason = load_curve(s["run_dir"])
        if len(ep) == 0:
            continue
        lbl = f"{s['optimizer']} lr={s['lr']}"
        if s["dropout"] > 0:
            lbl += f" do={s['dropout']}"
        if s["bidirectional"]:
            lbl += " bidir"
        if reason != "converged":
            lbl += "*"
        ax.plot(ep, err, color=colors[i % len(colors)], linewidth=1.7, label=lbl)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Training Error")
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage1_heatmap(summaries: List[Dict[str, Any]], title: str, out_path: Path) -> None:
    s1 = [s for s in summaries if s["stage"] == "stage1_arch"]
    if not s1:
        return
    layers = sorted({s["num_layers"] for s in s1})
    hiddens = sorted({s["hidden_size"] for s in s1})
    grid = np.full((len(layers), len(hiddens)), np.nan)
    lookup = {(s["num_layers"], s["hidden_size"]): s for s in s1}

    for i, nl in enumerate(layers):
        for j, hs in enumerate(hiddens):
            if (nl, hs) in lookup:
                grid[i, j] = lookup[(nl, hs)]["val_acc"] * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=220)
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="Val Accuracy (%)")
    ax.set_xticks(np.arange(len(hiddens)))
    ax.set_xticklabels([str(h) for h in hiddens])
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("Hidden Size (Neurons per Layer)")
    ax.set_ylabel("Number of Recurrent Layers")
    ax.set_title(title, fontsize=11)

    valid = grid[np.isfinite(grid)]
    thresh = (valid.min() + valid.max()) / 2.0 if len(valid) > 0 else 50.0
    for i, nl in enumerate(layers):
        for j, hs in enumerate(hiddens):
            s = lookup.get((nl, hs))
            if s:
                txt = f"Val: {s['val_acc'] * 100:.1f}%\nTest: {s['test_acc'] * 100:.1f}%"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white" if grid[i, j] > thresh else "black",
                )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_handwritten_sample_grids(
    data_root: Path,
    results_root: Path,
    invert_y: bool = INVERT_Y,
    samples_per_class: int = 6,
) -> None:
    _, class_names, train_records, _ = discover_dataset(data_root, "handwritten")
    out_dir = results_root / "handwritten"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_class: Dict[str, List[np.ndarray]] = {c: [] for c in class_names}
    for rec in train_records:
        if len(by_class[rec.class_name]) < samples_per_class:
            raw, _ = parse_handwritten_file(rec.path)
            by_class[rec.class_name].append(normalize_handwritten_xy(raw))

    fig, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(2.2 * samples_per_class, 2.2 * len(class_names)),
        dpi=220,
    )
    for row_idx, cls_name in enumerate(class_names):
        samples = by_class[cls_name]
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                render_stroke_on_axis(ax, samples[col_idx], invert_y=invert_y)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(cls_name, fontsize=11, fontweight="bold")
                if row_idx == 0:
                    ax.set_title(f"Sample {col_idx + 1}", fontsize=10)
            else:
                ax.axis("off")

    fig.suptitle(f"Normalized Handwritten Character Samples ([0, 1], invert_y={invert_y})", fontsize=12.5, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "handwritten_samples_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for cls_name in class_names:
        samples = by_class[cls_name]
        fig_c, axes_c = plt.subplots(1, len(samples), figsize=(2.3 * len(samples), 2.5), dpi=220)
        if len(samples) == 1:
            axes_c = [axes_c]
        for j, pts in enumerate(samples):
            render_stroke_on_axis(axes_c[j], pts, invert_y=invert_y, title=f"{cls_name} #{j + 1}")
        fig_c.tight_layout()
        fig_c.savefig(out_dir / f"handwritten_samples_{cls_name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig_c)


def generate_dataset_distribution_plot(data_root: Path, results_root: Path, dataset: str) -> None:
    _, class_names, train_records, test_records = discover_dataset(data_root, dataset)
    parser_fn = parse_handwritten_file if dataset == "handwritten" else parse_cv_file

    tr_counts = [sum(1 for r in train_records if r.class_name == c) for c in class_names]
    te_counts = [sum(1 for r in test_records if r.class_name == c) for c in class_names]
    tr_lens = [parser_fn(r.path)[0].shape[0] for r in train_records]
    te_lens = [parser_fn(r.path)[0].shape[0] for r in test_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=220)
    x = np.arange(len(class_names))
    w = 0.36
    ax1.bar(x - w / 2, tr_counts, width=w, label="Train (+Val)", color="#1f77b4")
    ax1.bar(x + w / 2, te_counts, width=w, label="Test", color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.set_ylabel("Samples")
    ax1.set_title(f"{dataset}: Class Balance")
    ax1.legend()
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax2.hist(tr_lens, bins=20, alpha=0.65, label=f"Train (mean={np.mean(tr_lens):.1f})", color="#1f77b4")
    ax2.hist(te_lens, bins=20, alpha=0.65, label=f"Test (mean={np.mean(te_lens):.1f})", color="#ff7f0e")
    ax2.set_xlabel("Sequence Length (Time Steps)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{dataset}: Sequence Length Distribution")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path = results_root / dataset / "dataset_distributions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_dataset_comparisons(results_root: Path, dataset: str) -> Dict[str, Any]:
    ds_dir = results_root / dataset
    digest: Dict[str, Any] = {"dataset": dataset}

    rnn_all_p, lstm_all_p = ds_dir / "rnn" / "all_configs.json", ds_dir / "lstm" / "all_configs.json"
    if rnn_all_p.is_file() and lstm_all_p.is_file():
        rnn_s1 = {(s["num_layers"], s["hidden_size"]): s for s in load_json(rnn_all_p) if s["stage"] == "stage1_arch"}
        lstm_s1 = {(s["num_layers"], s["hidden_size"]): s for s in load_json(lstm_all_p) if s["stage"] == "stage1_arch"}
        paired_rows = []
        for nl, hs in sorted(set(rnn_s1.keys()) & set(lstm_s1.keys())):
            r, l = rnn_s1[(nl, hs)], lstm_s1[(nl, hs)]
            paired_rows.append(
                {
                    "num_layers": nl,
                    "hidden_size": hs,
                    "rnn_val_acc": r["val_acc"],
                    "lstm_val_acc": l["val_acc"],
                    "lstm_minus_rnn_val": l["val_acc"] - r["val_acc"],
                    "rnn_test_acc": r["test_acc"],
                    "lstm_test_acc": l["test_acc"],
                    "lstm_minus_rnn_test": l["test_acc"] - r["test_acc"],
                    "rnn_epochs": r["epochs"],
                    "lstm_epochs": l["epochs"],
                }
            )
        paired_df = pd.DataFrame(paired_rows)
        if not paired_df.empty:
            save_csv_and_md(paired_df, ds_dir / "rnn_vs_lstm_stage1_paired")
            digest["n_stage1_paired"] = int(len(paired_df))
            digest["mean_stage1_lstm_minus_rnn_val"] = float(paired_df["lstm_minus_rnn_val"].mean())
            digest["mean_stage1_lstm_minus_rnn_test"] = float(paired_df["lstm_minus_rnn_test"].mean())

            fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=220)
            labels = [f"L{int(row['num_layers'])}-H{int(row['hidden_size'])}" for _, row in paired_df.iterrows()]
            x = np.arange(len(labels))
            w = 0.36
            ax.bar(x - w / 2, paired_df["rnn_epochs"], width=w, label="RNN", color="#1f77b4")
            ax.bar(x + w / 2, paired_df["lstm_epochs"], width=w, label="LSTM", color="#2ca02c")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Epochs to Converge")
            ax.set_title(f"{dataset}: Stage 1 Epochs to Convergence (RNN vs LSTM)")
            ax.legend()
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(ds_dir / "epochs_to_converge.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    rnn_best_p, lstm_best_p = ds_dir / "rnn" / "best_model_details.json", ds_dir / "lstm" / "best_model_details.json"
    if rnn_best_p.is_file() and lstm_best_p.is_file():
        rb, lb = load_json(rnn_best_p), load_json(lstm_best_p)
        splits = ["Train", "Val", "Test"]
        r_vals = [rb["train_acc"] * 100, rb["val_acc"] * 100, rb["test_acc"] * 100]
        l_vals = [lb["train_acc"] * 100, lb["val_acc"] * 100, lb["test_acc"] * 100]

        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=220)
        x = np.arange(len(splits))
        w = 0.34
        b1 = ax.bar(x - w / 2, r_vals, width=w, label="Best RNN", color="#1f77b4")
        b2 = ax.bar(x + w / 2, l_vals, width=w, label="Best LSTM", color="#2ca02c")
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{dataset}: Best RNN vs Best LSTM")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(ds_dir / "rnn_vs_lstm_comparison.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    return digest


def update_summary_markdown(results_root: Path, digests: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    digests = dict(digests or {})
    lines = ["# Assignment 6 Results Summary\n"]
    for ds_name in ("handwritten", "cv"):
        dg = digests.get(ds_name)
        if dg is None:
            rnn_p = results_root / ds_name / "rnn" / "best_model_details.json"
            lstm_p = results_root / ds_name / "lstm" / "best_model_details.json"
            if rnn_p.is_file() or lstm_p.is_file():
                dg = generate_dataset_comparisons(results_root=results_root, dataset=ds_name)
                digests[ds_name] = dg
        if not dg:
            continue
        lines.append(f"## Dataset: `{ds_name}`\n")
        if "mean_stage1_lstm_minus_rnn_val" in dg:
            lines.append(
                f"- **Stage 1 Paired Comparison (LSTM - RNN across {dg.get('n_stage1_paired', 0)} architectures):** "
                f"Val delta = `{dg['mean_stage1_lstm_minus_rnn_val']*100:+.2f}%`, "
                f"Test delta = `{dg['mean_stage1_lstm_minus_rnn_test']*100:+.2f}%`"
            )
        for m_type in ("rnn", "lstm"):
            best_p = results_root / ds_name / m_type / "best_model_details.json"
            all_p = results_root / ds_name / m_type / "all_configs.json"
            if best_p.is_file():
                b = load_json(best_p)
                cfg = b["config"]
                lines.append(
                    f"- **Best `{m_type.upper()}`** (`{b['stage']}`): "
                    f"`hidden={cfg['hidden_size']}, layers={cfg['num_layers']}, bi={cfg['bidirectional']}, "
                    f"do={cfg['dropout']}, opt={cfg['optimizer']}, lr={cfg['lr']}` | params=`{b['n_params']}`"
                )
                lines.append(
                    f"  - Train=`{b['train_acc']*100:.2f}%`, Val=`{b['val_acc']*100:.2f}%`, "
                    f"Test=`{b['test_acc']*100:.2f}%`, Epochs=`{b['epochs']}` ({b['stop_reason']})"
                )
            if all_p.is_file():
                all_cfgs = load_json(all_p)
                non_conv = sum(1 for c in all_cfgs if not c.get("converged", False))
                div_cnt = sum(1 for c in all_cfgs if c.get("diverged", False))
                lines.append(
                    f"  - `{m_type.upper()}` sweep: `{len(all_cfgs)}` configs, `{non_conv}` non-converged, `{div_cnt}` diverged."
                )
        lines.append("")
    (results_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all_plots_for_dataset(data_root: Path, results_root: Path, dataset: str) -> Dict[str, Any]:
    if dataset == "handwritten":
        export_all_handwritten_normalized_images(data_root=data_root, invert_y=INVERT_Y, overwrite=False)
        generate_handwritten_sample_grids(data_root=data_root, results_root=results_root, invert_y=INVERT_Y)

    try:
        generate_dataset_distribution_plot(data_root=data_root, results_root=results_root, dataset=dataset)
    except Exception:
        pass

    for m_type in ("rnn", "lstm"):
        model_dir = results_root / dataset / m_type
        all_p = model_dir / "all_configs.json"
        if all_p.is_file():
            summaries = load_json(all_p)
            for idx, s in enumerate(summaries):
                plot_single_error_curve(
                    s["run_dir"],
                    title=f"{dataset}/{m_type}: {s['run_id']}",
                    out_path=model_dir / "error_curves" / f"{idx + 1:02d}_{s['stage']}_{s['run_id']}.png",
                )
            plot_stage1_overview(summaries, f"{dataset} / {m_type.upper()}: Stage 1 Training Error", model_dir / "stage1_error_curves.png")
            plot_stage2_overview(summaries, f"{dataset} / {m_type.upper()}: Stage 2 Tuning Error", model_dir / "stage2_error_curves.png")
            plot_stage1_heatmap(summaries, f"{dataset} / {m_type.upper()}: Stage 1 Val Accuracy Heatmap", model_dir / "stage1_heatmap.png")

        best_p = model_dir / "best_model_details.json"
        if best_p.is_file():
            details = load_json(best_p)
            cls_names = details["class_names"]
            if details.get("test_cm"):
                plot_confusion_matrix(
                    np.asarray(details["test_cm"]),
                    cls_names,
                    title=f"{dataset} / {m_type.upper()} Test CM (Acc={details['test_acc']*100:.2f}%)",
                    save_path=model_dir / "cm_best_test_counts.png",
                )
                plot_confusion_matrix(
                    np.asarray(details["test_cm"]),
                    cls_names,
                    title=f"{dataset} / {m_type.upper()} Normalized Test CM",
                    save_path=model_dir / "cm_best_test_normalized.png",
                    normalized=True,
                )
            if details.get("train_cm"):
                plot_confusion_matrix(
                    np.asarray(details["train_cm"]),
                    cls_names,
                    title=f"{dataset} / {m_type.upper()} Train CM (Acc={details['train_acc']*100:.2f}%)",
                    save_path=model_dir / "cm_best_train_counts.png",
                )

    digest = generate_dataset_comparisons(results_root=results_root, dataset=dataset)
    return digest
