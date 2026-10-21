# CS601T Assignment 5 — Code Review & Fixes

Files reviewed: `part_a.py`, `part_b.py`, `utils/metrics.py`, `utils/plotting.py`, `utils/receptive_field.py`, `utils/select_images.py`, against `CS601T_Assignment5_Details_12October2026.pdf`.

> **Testing note:** the snippets below were written against the assignment spec and the library APIs, but I could not run them (no dataset/GPU). The PyTorch snippets are small and low-risk. The TensorFlow/Keras script in §B-4 is the largest piece. Run it once on a few images before you trust the numbers, and check the first Grad-CAM output looks different for different classes.

---

## 0. Summary table

| # | Sev. | File | Issue |
|---|------|------|-------|
| B-4a | 🔴 Bug | `part_b.py` | Grad-CAM uses mean **activations**, not **gradients**, so every class gets the identical heatmap |
| B-4b | 🔴 Bug | `part_b.py` | Guided Backprop hooks never attach (Keras VGG19 ReLU is inside `Conv2D`), so output is plain saliency |
| B-4c | 🔴 Bug | `part_b.py` | `CrossEntropyLoss` applied to a softmax output (double softmax) |
| B-4d | 🔴 Spec | `part_b.py` | Assignment requires **TensorFlow Keras**. Code uses Keras-on-torch with a PyTorch loop (and `torch.optim.Adam(model.trainable_variables)` is not the supported way) |
| B-4e | 🟠 Bug | `part_b.py` | `keras.Model(model.inputs, [base_vgg.get_layer(...).output, ...])` is a disconnected graph (layer lives in the nested model) |
| B-4f | 🟠 Spec | `part_b.py` | VGG head drops pretrained fc1/fc2; should replace only the final 1000-way layer |
| A-1 | 🔴 Missing | `part_a.py` | Q2: filters never plotted; pooled outputs never shown; feature maps not "printed" |
| A-2 | 🔴 Missing | `part_a.py` | Q3b: no filter images; only 4 of 10 conv2 feature maps |
| B-1 | 🔴 Missing | `part_b.py` | Q1a: no per-layer feature-map size description |
| B-2 | 🔴 Missing | `part_b.py` | Q1b: best-architecture **test** confusion matrix not plotted |
| B-3 | 🔴 Missing | `part_b.py` | Q1c: 8 maps required from *each* remaining conv layer; only last layer plotted |
| A-3 | 🟠 Spec | `part_a.py` | Q1/Q2: expected dims hard-coded rather than computed from the formula; no manual traversal |
| G-1 | 🟠 | both | No random seeds, so report numbers are not reproducible |
| A-4 | 🟡 | `part_a.py` | "SGD" uses momentum + weight decay (state it or make it optional) |
| G-2 | 🟡 | `utils/metrics.py` | `confusion_matrix` is a slow Python loop, and mixes CPU/GPU tensors |
| B-5 | 🟡 | `part_b.py` | VGG19 analytic RF = 252 > 224, so boxes cover the whole image (explain in report / add gradient-based patch) |
| G-3 | 🟡 | repo | Dead code, packaging, deadline date mismatch |

🔴 = wrong result or required deliverable missing · 🟠 = deviates from spec / likely failure · 🟡 = polish

---

## G-1. Reproducibility (both files)

No seeds anywhere, so the filter values and accuracies in your report will not match a re-run.

Add to `utils/seed.py`:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Call it at the top of `main()` in both `part_a.py` and `part_b.py`:

```python
from utils.seed import set_seed
...
def main():
    ...
    set_seed(42)
```

(For the TF script, `keras.utils.set_random_seed(42)` is used in §B-4.)

---

## G-2. `utils/metrics.py` — vectorised confusion matrix

The current version loops per sample in Python and indexes a CPU tensor with CUDA scalars. Replace `confusion_matrix`:

```python
def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Confusion matrix (rows = true class, cols = predicted class)."""
    p = predictions.reshape(-1).long().cpu()
    y = labels.reshape(-1).long().cpu()
    return torch.bincount(y * num_classes + p, minlength=num_classes ** 2).view(num_classes, num_classes)
```

Also move the duplicated `evaluate_split` (identical in `part_a.py` and `part_b.py`) into `utils/metrics.py` and import it in both.

---

# PART A

## A-3. Q1 / Q2 — compute expected dimensions from the formula + manual traversal

**Problem:** `part_a.py` hard-codes `(input_h - 5) // 1 + 1` and a hard-coded dict of expected shapes. The brief says "calculate using the mathematical formula" and "traverse the convolution filter over all the pixels". Using only `nn.Conv2d` may not satisfy "traverse" if your instructor wants it spelled out.

Add near the top of `part_a.py` (and `import torch.nn.functional as F`):

```python
def conv_out_size(n: int, k: int, s: int = 1, p: int = 0) -> int:
    """W_out = floor((W_in - K + 2P) / S) + 1"""
    return (n - k + 2 * p) // s + 1


def manual_conv2d(img: torch.Tensor, kernel: torch.Tensor, bias: float = 0.0,
                  stride: int = 1, padding: int = 0) -> torch.Tensor:
    """Slide a k x k kernel over a single-channel (H, W) image (cross-correlation, as in nn.Conv2d)."""
    if padding:
        img = F.pad(img, (padding,) * 4)
    k = kernel.shape[-1]
    H, W = img.shape
    out_h, out_w = (H - k) // stride + 1, (W - k) // stride + 1
    out = torch.zeros(out_h, out_w)
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i * stride:i * stride + k, j * stride:j * stride + k]
            out[i, j] = (patch * kernel).sum() + bias
    return out


def expected_stack_shapes(h: int = 28, w: int = 28) -> dict:
    c1 = conv_out_size(h, 5, 1, 0)
    p1 = conv_out_size(c1, 2, 2, 0)
    c2 = conv_out_size(p1, 5, 1, 0)
    p2 = conv_out_size(c2, 2, 2, 0)
    return {"conv1": (6, c1, c1), "pool1": (6, p1, p1), "conv2": (16, c2, c2), "pool2": (16, p2, p2)}
```

**In `single_filter_convolution_analysis`**, replace the two `expected_*` lines and add a manual check right after `feature_maps = conv(...)`:

```python
    input_h, input_w = images.shape[2], images.shape[3]
    expected_h = conv_out_size(input_h, 5, stride=1, p=0)
    expected_w = conv_out_size(input_w, 5, stride=1, p=0)
    ...
    # verify the library output against an explicit sliding-window implementation
    k_cpu = conv.weight.detach().cpu()[0, 0]
    b_cpu = conv.bias.detach().cpu()[0]
    manual = torch.stack([manual_conv2d(images[i, 0], k_cpu, b_cpu) for i in range(len(images))]).unsqueeze(1)
    assert torch.allclose(manual, feature_maps, atol=1e-5), "manual conv != nn.Conv2d"
    print("Manual sliding-window convolution matches nn.Conv2d")
```

**In `stacked_convolution_analysis`**, replace the hard-coded dict:

```python
    expected_shapes = expected_stack_shapes(images.shape[2], images.shape[3])
```

---

## A-1. Q2 — filters, pooled outputs, and "print the feature map"

**Problems in `stacked_convolution_analysis`:**
1. The three conv1 / three conv2 **filters are never drawn** (only saved as JSON).
2. Only pre-pool maps (24×24, 8×8) are plotted, so the output *after each layer* (12×12, 4×4) is never shown.
3. The brief says to "print the feature map obtained after each layer"; only shapes are printed.

### Step 1: append these helpers to `utils/plotting.py`

```python
def plot_filter_images(conv1_w, conv2_w, idx1, idx2, path, title="Convolutional filters"):
    """Draw selected conv1 / conv2 kernels. conv2 kernels (6x5x5) are averaged over input channels."""
    conv1_w, conv2_w = conv1_w.detach().cpu(), conv2_w.detach().cpu()
    n = max(len(idx1), len(idx2))
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.6), squeeze=False)
    for row, (w, idxs, name) in enumerate([(conv1_w, idx1, "Conv1"), (conv2_w, idx2, "Conv2")]):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(idxs):
                kernel = w[idxs[col]].mean(dim=0).numpy()
                ax.imshow(kernel, cmap="viridis")
                suffix = "" if row == 0 else "\n(mean over in-ch)"
                ax.set_title(f"{name} filter {idxs[col] + 1}{suffix}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_map_grid(images, class_names, maps1, maps2, idx1, idx2, path,
                          title="Feature maps", stage_names=("Layer 1", "Layer 2")):
    """One row per input image: [input | maps1[idx1...] | maps2[idx2...]]. maps are (N, C, H, W) CPU tensors."""
    n_img = images.shape[0]
    ncol = 1 + len(idx1) + len(idx2)
    fig, axes = plt.subplots(n_img, ncol, figsize=(1.7 * ncol, 1.9 * n_img), squeeze=False)
    for r in range(n_img):
        axes[r, 0].imshow(images[r, 0].cpu().numpy(), cmap="gray")
        axes[r, 0].set_title(f"'{class_names[r]}'", fontsize=8)
        for j, f in enumerate(idx1):
            ax = axes[r, 1 + j]
            ax.imshow(maps1[r, f].numpy(), cmap="magma")
            if r == 0:
                ax.set_title(f"{stage_names[0]}\nF{f + 1} ({maps1.shape[2]}x{maps1.shape[3]})", fontsize=7)
        for j, f in enumerate(idx2):
            ax = axes[r, 1 + len(idx1) + j]
            ax.imshow(maps2[r, f].numpy(), cmap="magma")
            if r == 0:
                ax.set_title(f"{stage_names[1]}\nF{f + 1} ({maps2.shape[2]}x{maps2.shape[3]})", fontsize=7)
        for c in range(ncol):
            axes[r, c].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
```

### Step 2: in `part_a.py` change the import

```python
from utils.plotting import plot_training_curves, plot_filter_images, plot_feature_map_grid
```

### Step 3: replace the plotting block + records in `stacked_convolution_analysis`

Delete everything from `conv1_w = ...` through the `(RESULTS_DIR / "stacked_convolution_observations.json")` write, and use:

```python
    conv1_w = model.conv1.weight.detach().cpu()   # (6, 1, 5, 5)
    conv2_w = model.conv2.weight.detach().cpu()   # (16, 6, 5, 5)
    sel = [0, 1, 2]                               # ANY 3 filters; keep these indices for Q3b

    # 3 filters per layer as images
    plot_filter_images(conv1_w, conv2_w, sel, sel,
                       RESULTS_DIR / "stacked_filters.png",
                       "Q2: selected filters (random Kaiming init)")

    # feature maps AFTER ReLU (before pooling) and AFTER each full layer (after pooling)
    plot_feature_map_grid(images, class_names, out["relu1"].cpu(), out["relu2"].cpu(), sel, sel,
                          RESULTS_DIR / "stacked_feature_maps_after_relu.png",
                          "Q2: feature maps after conv + ReLU (24x24 / 8x8)",
                          stage_names=("Conv1+ReLU", "Conv2+ReLU"))
    plot_feature_map_grid(images, class_names, out["pool1"].cpu(), out["pool2"].cpu(), sel, sel,
                          RESULTS_DIR / "stacked_feature_maps_after_pool.png",
                          "Q2: feature maps after full layer = conv + ReLU + maxpool (12x12 / 4x4)",
                          stage_names=("Layer1 out", "Layer2 out"))

    # "print the feature map obtained after each layer"
    torch.set_printoptions(precision=3, linewidth=160, sci_mode=False)
    for name in ("pool1", "pool2"):
        print(f"\n{name} output, image of class '{class_names[0]}', channel 1:\n{out[name][0, 0].cpu()}")
    np.savez(RESULTS_DIR / "stacked_feature_maps.npz", **{k: v.cpu().numpy() for k, v in out.items()})

    records = {
        "classes": class_names,
        "selected_filter_indices_0based": sel,
        "conv1_shape": list(out["conv1"].shape),
        "pool1_shape": list(out["pool1"].shape),
        "conv2_shape": list(out["conv2"].shape),
        "pool2_shape": list(out["pool2"].shape),
        "conv1_filters_selected": conv1_w[sel, 0].tolist(),
        "conv2_filters_selected_mean_over_inchannels": conv2_w[sel].mean(dim=1).tolist(),
    }
    (RESULTS_DIR / "stacked_convolution_observations.json").write_text(json.dumps(records, indent=2))
    plot_path = RESULTS_DIR / "stacked_feature_maps_after_pool.png"
    print(f"Saved stacked convolution figures to {RESULTS_DIR}")
```

---

## A-2. Q3b — filter images and 10 feature maps per layer

**Problems (`train_classifier_and_visualize_features`, the "Feature Map and Filter Weight Visualizations" block):**
1. Filter weights are not drawn at all (the figure title says "Learned Convolutional Filters" but it only shows feature maps).
2. Only 4 conv2 feature maps are plotted; the brief asks for 10.
3. Layer 1 has **only 6 filters**, so "10 filters from the first layer" cannot be met literally. Plot all 6 and **state this in the report** (or confirm with the TA).
4. Q3b uses pre-ReLU maps (`feat["conv1"]`) while Q2 plots post-ReLU maps. Make them consistent.
5. "Same position/location index as Q2": Q2 used filters 0, 1, 2; `range(10)` includes them.

Replace the whole block from `with torch.no_grad(): feat = ...` down to (and including) `plt.close(fig)` of the filters figure, i.e. keep only the line that defines `feat` / `c2_fmaps` if you still need `conv2_act` below:

```python
    # ========================================================================
    # Q3b: filters + feature maps of the TRAINED network
    # ========================================================================
    model.eval()
    with torch.no_grad():
        feat = model.extract_intermediate_features(images.to(device))

    idx1 = list(range(6))    # conv1 has only 6 filters (assignment says 10; see report note)
    idx2 = list(range(10))   # first 10 of conv2's 16 filters; includes the Q2 indices 0, 1, 2

    plot_filter_images(model.conv1.weight, model.conv2.weight, idx1, idx2,
                       RESULTS_DIR / "trained_filters.png",
                       "Q3b: trained filters (conv2 kernels averaged over 6 input channels)")

    plot_feature_map_grid(images, class_names, feat["relu1"].cpu(), feat["relu2"].cpu(), idx1, idx2,
                          RESULTS_DIR / "trained_feature_maps.png",
                          "Q3b: feature maps of the trained CNN (after ReLU, before pooling)",
                          stage_names=("Conv1+ReLU", "Conv2+ReLU"))
```

> The receptive-field section below it uses `feat["conv2"]`. That is fine (last conv layer, 8×8, RF = 14×14); keep it. Mention in the report that "last convolutional layer" was taken *before* pooling.

---

## A-4. "SGD" with momentum and weight decay

The brief says "backpropagation with stochastic gradient descent". You use `momentum=0.9, weight_decay=1e-4` and mini-batches of 64. That is defensible, but say so in the report, or make it switchable:

```python
parser.add_argument("--momentum", type=float, default=0.9, help="0 = vanilla SGD")
...
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=momentum, weight_decay=1e-4)
```

(Pass `args.momentum` through `train_classifier_and_visualize_features`.) Also note in the report that the model outputs logits and `nn.CrossEntropyLoss` applies the softmax internally.

---

# PART B

## B-0. Split the file: remove all Keras code from `part_b.py`

`part_b.py` should become a **pure PyTorch** script (custom CNNs, Q1). VGG19 (Q2) moves to a new TensorFlow script `part_b_vgg.py` (§B-4). Mixing `KERAS_BACKEND=torch` with `import tensorflow` in one process will not work.

In `part_b.py`:

- **Delete** the lines `os.environ.setdefault("KERAS_BACKEND", "torch")`, `import keras`, `from keras import layers`, `from keras.applications import vgg19 as keras_vgg19`, `import matplotlib.cm as cm`, and `import os`.
- **Delete** `build_vgg19_classifier_keras`, `blend_heatmap_with_image`, `create_guided_backprop_overlay`, `vgg19_transfer_learning_and_interpretability`, `GRADCAM_DIR`, `GUIDED_BACKPROP_DIR`.
- **Replace** `get_device`:

```python
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    print(f"Active compute device: {device}{gpu_name}")
    return device
```

- **Replace** `main`:

```python
def main():
    parser = argparse.ArgumentParser(description="Part B Q1: custom CNN architectures on Caltech-101")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    device = get_device()
    loaders, class_to_idx, idx_to_class = load_caltech_dataloaders(batch_size=16, data_dir=CALTECH_DIR)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    images, _, _, _ = sample_representative_images(split="train", data_dir=CALTECH_DIR)

    compare_custom_architectures(loaders, class_names, images, epochs=args.epochs, device=device)
```

and add `from utils.seed import set_seed` to the imports.

---

## B-1. Q1a — describe feature-map sizes at each layer

**Problem:** the brief says "clearly describe the architectures about the size of the resulting feature maps at each layer". Nothing prints or saves this (only a hard-coded `(54x54)` label).

Add to `part_b.py`:

```python
def describe_architecture(model: HierarchicalCaltechCNN, device: torch.device) -> str:
    """Return (and print) a markdown table of layer -> output shape -> #params."""
    x = torch.zeros(1, 3, 224, 224, device=device)
    rows = [("Input", "-", tuple(x.shape[1:]), 0)]
    conv_i = 0
    model.eval()
    with torch.no_grad():
        for m in model.features:
            x = m(x)
            if isinstance(m, nn.Conv2d):
                conv_i += 1
                k, s, p = m.kernel_size[0], m.stride[0], m.padding[0]
                rows.append((f"Conv{conv_i}: {m.out_channels}@{k}x{k}, s={s}, p={p} + ReLU", "",
                             tuple(x.shape[1:]), sum(q.numel() for q in m.parameters())))
            elif isinstance(m, nn.MaxPool2d):
                rows.append((f"MaxPool{conv_i}: 3x3, s=2", "", tuple(x.shape[1:]), 0))
        flat = x.flatten(1).shape[1]
    fc1, fc2 = model.classifier[1], model.classifier[3]
    rows.append(("Flatten", "", (flat,), 0))
    rows.append((f"FC1: {fc1.out_features} + ReLU", "", (fc1.out_features,), sum(q.numel() for q in fc1.parameters())))
    rows.append((f"FC2: {fc2.out_features} (softmax via loss)", "", (fc2.out_features,), sum(q.numel() for q in fc2.parameters())))

    md = ["| Layer | Output shape (C,H,W) | Params |", "|---|---|---|"]
    md += [f"| {name} | {shape} | {n:,} |" for name, _, shape, n in rows]
    md.append(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    table = "\n".join(md)
    print(f"\n{model.architecture.upper()}\n{table}")
    return table
```

Call it for every architecture inside `compare_custom_architectures`, right after `model = HierarchicalCaltechCNN(...).to(device)`:

```python
        (RESULTS_DIR / f"{arch}_architecture.md").write_text(describe_architecture(model, device))
```

Expected (I checked these by hand/script): Arch1 `54→26→22→10`, flatten 1600; Arch2 `…→8→3`, flatten 288; Arch3 `…→8→6→2`, flatten 256. Put this table in the report.

---

## B-2. Q1b — plot the best architecture's test confusion matrix

**Problem:** the test CM is stored in JSON only. Add the import at the top of `part_b.py`:

```python
from utils.metrics import accuracy, confusion_matrix, plot_confusion_matrix
```

and right after `results[best_arch]["test_cm"] = ...`:

```python
    plot_confusion_matrix(
        test_cm, class_names,
        f"{best_arch.upper()} - Test set (Acc: {test_acc * 100:.1f}%)",
        RESULTS_DIR / "best_architecture_test_confusion_matrix.png",
    )
```

---

## B-3. Q1c — 8 feature maps from each remaining conv layer

**Problem:** the code plots 8 maps from conv1 and 8 maps from only the **last** conv layer (`fmaps[-1]`). The brief wants "a selected 8 feature maps from the **remaining convolutional layers**" (i.e. conv2, conv3, …).

Add to `part_b.py`:

```python
def plot_layerwise_feature_maps(fmaps, input_img, path, title, n_maps: int = 8):
    """fmaps: list of (1, C, H, W) tensors (one per conv layer). Row i = conv layer i+1."""
    n_layers = len(fmaps)
    fig, axes = plt.subplots(n_layers + 1, n_maps, figsize=(2.1 * n_maps, 2.3 * (n_layers + 1)), squeeze=False)

    for c in range(n_maps):                      # row 0: the input image (shown once)
        axes[0, c].axis("off")
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Input image", fontsize=9)

    chosen = {}
    for li, fm in enumerate(fmaps, start=1):
        fm = fm[0].detach().cpu().numpy()        # (C, H, W)
        C, H, W = fm.shape
        idx = np.arange(C) if C <= n_maps else np.linspace(0, C - 1, n_maps, dtype=int)  # conv1 has exactly 8
        chosen[f"conv{li}"] = [int(i) + 1 for i in idx]
        for j in range(n_maps):
            ax = axes[li, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(fm[idx[j]], cmap="magma")
                ax.set_title(f"Conv{li} ch{idx[j] + 1}\n{H}x{W}", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return chosen
```

In `compare_custom_architectures`, replace the whole "Feature Map Comparison: Shallow vs. Deep Representations" block with:

```python
    # ========================================================================
    # Q1c: feature maps of ONE training image across ALL conv layers
    # ========================================================================
    single_img = images[0:1].to(device)
    with torch.no_grad():
        fmaps = best_model.extract_conv_feature_maps(single_img)   # post-ReLU, one per conv layer
    chosen = plot_layerwise_feature_maps(
        fmaps, np.transpose(images[0].numpy(), (1, 2, 0)),
        RESULTS_DIR / "best_architecture_feature_maps.png",
        f"{best_arch.upper()}: 8 feature maps per conv layer (class '{class_names[0]}')",
    )
    print("Channels plotted per layer (1-based):", chosen)
```

---

## B-3b. (Optional but recommended) Q1d — also show the cropped patch

Q1d says "visualize the patches". Boxes are drawn already; adding the crop makes the figure self-explanatory. In the Q1d loop of `compare_custom_architectures` replace the 1×5 figure with a 2×5 one:

```python
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    for idx in range(5):
        cname = class_names[idx]
        img_np = np.transpose(images[idx].numpy(), (1, 2, 0))
        act = last_act[idx]
        ch, r, c = np.unravel_index(np.argmax(act), act.shape)
        val = float(act[ch, r, c])
        ymin, xmin, ymax, xmax = neuron_to_bounding_box(int(r), int(c), rf, jump, start, image_shape=(224, 224))
        patch_results[cname] = {"channel": int(ch), "row": int(r), "col": int(c), "val": val, "bbox": [ymin, xmin, ymax, xmax]}

        axes[0, idx].imshow(img_np)
        axes[0, idx].add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                                 linewidth=2.5, edgecolor="#ff3333", facecolor="none"))
        axes[0, idx].set_title(f"'{cname}'  ch{ch} @({r},{c})\nact={val:.2f}", fontsize=9)
        axes[1, idx].imshow(img_np[ymin:ymax, xmin:xmax])
        axes[1, idx].set_title(f"Patch {ymax - ymin}x{xmax - xmin}", fontsize=9)
        axes[0, idx].axis("off"); axes[1, idx].axis("off")
```

(The analytic RF code above this is correct. For Arch1 the RF is 51×51; it is computed with the final pool excluded, which matches `fmaps[-1]` being pre-pool.)

---

## B-4. Q2 — VGG19 in TensorFlow Keras (replacement script)

### What was wrong

**B-4a Grad-CAM uses activations instead of gradients.**
```python
pooled_grads = conv_out.mean(dim=(1, 2), keepdim=True)   # <- mean of ACTIVATIONS
```
`score.backward()` is called but `.grad` is never read, so the heatmap does not depend on `cls_i`. All 5 "target class" heatmaps for one image come out identical, which defeats the "its class and other classes" requirement. Grad-CAM must use `mean(∂score/∂conv_out)`. Also use the pre-softmax **logit** as the score to avoid saturated gradients.

**B-4b Guided Backprop hooks never attach.** In Keras, VGG19 is built as `Conv2D(..., activation="relu")`. The ReLU is *inside* the conv layer, not a separate layer. The loop looking for `nn.ReLU` matches nothing, so `hooks == []` and the result is ordinary gradient saliency. Guided backprop requires replacing the ReLU's gradient with `1[x>0] · 1[dy>0] · dy`.

**B-4c Double softmax.** The head is `Dense(5, activation="softmax")` but the loss is `nn.CrossEntropyLoss()`, which applies `log_softmax` again. Output logits and use `from_logits=True`.

**B-4d Not TensorFlow Keras.** The brief says "Leverage Tensorflow Keras API". The code uses `KERAS_BACKEND=torch` and a PyTorch optimiser. `torch.optim.Adam(model.trainable_variables)` passes Keras `Variable` wrappers (the supported call is `model.parameters()`). Using TF Keras avoids this entirely.

**B-4e Disconnected graph.** `keras.Model(model.inputs, [base_vgg.get_layer("block5_conv4").output, ...])` fails because `block5_conv4` belongs to the nested `base_vgg` graph. Build one flat functional model from `base.input` instead.

**B-4f Head design.** `include_top=False` + `Flatten` + `Dense(5)` throws away pretrained fc1/fc2. The brief says *modify the classification layer to 5 outputs and retrain only that layer*: keep fc1/fc2 frozen and swap only the last 1000-way layer.

### New file: `part_b_vgg.py`

Install: `pip install tensorflow pillow matplotlib numpy` (CPU TensorFlow is fine for a frozen backbone; native-Windows TF has no GPU support after 2.10).
Run **after** `part_b.py` so the comparison with Q1b can be written: `python part_b_vgg.py --epochs 10`.

```python
"""Part B Q2 — VGG19 transfer learning, Guided Backprop and Grad-CAM (TensorFlow Keras)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.metrics import plot_confusion_matrix
from utils.receptive_field import calculate_receptive_field, neuron_to_bounding_box

ROOT = Path(__file__).resolve().parent
CALTECH_DIR = ROOT / "data" / "caltech101_resized"
if not CALTECH_DIR.exists():
    CALTECH_DIR = ROOT / "data" / "caltech101_subset"
RESULTS_DIR = ROOT / "results" / "partB"
VGG_DIR = RESULTS_DIR / "vgg19"
GRADCAM_DIR = VGG_DIR / "gradcam"
GB_DIR = VGG_DIR / "guided_backprop"

IMG_SIZE = (224, 224)
BATCH = 16
SEED = 42
NUM_CLASSES = 5
preprocess_input = keras.applications.vgg19.preprocess_input


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_ds(split: str, shuffle: bool):
    """tf.data pipeline; class order is alphabetical (same as torchvision ImageFolder)."""
    ds = keras.utils.image_dataset_from_directory(
        CALTECH_DIR / split, labels="inferred", label_mode="int",
        image_size=IMG_SIZE, batch_size=BATCH, shuffle=shuffle, seed=SEED,
    )
    names = ds.class_names
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), names


def load_representative_images(split: str = "train"):
    """First image (sorted filename) of each class, identical selection to Part B Q1d / Part A."""
    class_dirs = sorted(d for d in (CALTECH_DIR / split).iterdir() if d.is_dir())
    arrs, paths = [], []
    for cdir in class_dirs:
        p = sorted(f for f in cdir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]
        arrs.append(keras.utils.img_to_array(keras.utils.load_img(p, target_size=IMG_SIZE)))
        paths.append(str(p))
    raw = np.stack(arrs).astype("float32")                      # (5, 224, 224, 3), 0..255, RGB
    return raw, preprocess_input(raw.copy()), paths             # copy: preprocess may work in place


# ----------------------------------------------------------------------------
# Model: frozen VGG19, only the new 5-way output layer is trained
# ----------------------------------------------------------------------------
def build_model() -> keras.Model:
    base = keras.applications.VGG19(weights="imagenet", include_top=True)   # conv base + fc1 + fc2 + predictions
    for layer in base.layers:
        layer.trainable = False
    logits = keras.layers.Dense(NUM_CLASSES, name="new_head")(base.layers[-2].output)  # replaces 1000-way layer
    model = keras.Model(base.input, logits, name="vgg19_5class")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def evaluate(model, ds):
    y_true, y_pred = [], []
    for x, y in ds:
        y_pred.append(np.argmax(model(x, training=False).numpy(), axis=1))
        y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return float((y_true == y_pred).mean()), cm


# ----------------------------------------------------------------------------
# Q2b: analytic receptive-field patches at block5_conv4
# ----------------------------------------------------------------------------
def vgg_receptive_field():
    blocks = [2, 2, 4, 4, 4]                       # conv layers per block (VGG19)
    specs = []
    for bi, n in enumerate(blocks):
        specs += [(3, 1, 1)] * n
        if bi < len(blocks) - 1:                   # no pool after block5: we stop at block5_conv4
            specs.append((2, 2, 0))
    return calculate_receptive_field(specs)        # -> RF = 252, jump = 16


def plot_rf_patches(conv_model, raw, pre, class_names, path):
    rf, jump, start = vgg_receptive_field()
    acts = conv_model(pre, training=False).numpy()               # (5, 14, 14, 512)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    info = {}
    for i in range(5):
        r, c, ch = np.unravel_index(np.argmax(acts[i]), acts[i].shape)
        ymin, xmin, ymax, xmax = neuron_to_bounding_box(int(r), int(c), rf, jump, start, image_shape=(224, 224))
        info[class_names[i]] = {"channel": int(ch), "row": int(r), "col": int(c),
                                "val": float(acts[i][r, c, ch]), "bbox": [ymin, xmin, ymax, xmax]}
        img = raw[i] / 255.0
        axes[0, i].imshow(img)
        axes[0, i].add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                               lw=2.5, ec="#00ffcc", fc="none"))
        axes[0, i].set_title(f"'{class_names[i]}' ch{ch} @({r},{c})", fontsize=9)
        axes[1, i].imshow(img[ymin:ymax, xmin:xmax])
        axes[1, i].set_title(f"patch {ymax - ymin}x{xmax - xmin}", fontsize=9)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    fig.suptitle(f"VGG19 block5_conv4: max-activated neuron per image (analytic RF = {int(rf)}x{int(rf)}, "
                 f"clipped to the 224x224 image)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    (VGG_DIR / "vgg19_rf_patch_info.json").write_text(json.dumps(info, indent=2))


# ----------------------------------------------------------------------------
# Q2c: Guided Backpropagation
# ----------------------------------------------------------------------------
@tf.custom_gradient
def guided_relu(x):
    y = tf.nn.relu(x)

    def grad(dy):
        return tf.cast(x > 0, dy.dtype) * tf.cast(dy > 0, dy.dtype) * dy
    return y, grad


class GuidedReluMode:
    """Temporarily swap every Conv2D's ReLU activation for guided_relu (restored on exit)."""
    def __init__(self, model):
        self.model, self.saved = model, {}

    def __enter__(self):
        for layer in self.model.layers:
            act = getattr(layer, "activation", None)
            if act is not None and getattr(act, "__name__", "") == "relu":
                self.saved[layer] = act
                layer.activation = guided_relu
        return self

    def __exit__(self, *exc):
        for layer, act in self.saved.items():
            layer.activation = act


def guided_backprop(conv_model, x_pre, ch, r, c):
    """Gradient of neuron (r, c, ch) of block5_conv4 w.r.t. the (preprocessed) input pixels."""
    x = tf.convert_to_tensor(x_pre[None] if x_pre.ndim == 3 else x_pre)
    with GuidedReluMode(conv_model):
        with tf.GradientTape() as tape:
            tape.watch(x)
            neuron = conv_model(x, training=False)[0, r, c, ch]
        g = tape.gradient(neuron, x)
    return g[0].numpy()


def gb_to_rgb(g):
    g = g[..., ::-1]                                  # preprocessed input is BGR -> RGB
    g = (g - g.mean()) / (g.std() + 1e-8)
    return np.clip(g * 0.25 + 0.5, 0.0, 1.0)


def run_guided_backprop(conv_model, raw, pre, class_names):
    """Use the SAME image as Q1c (first class image). Pick the 5 most-activated channels' peak neurons."""
    GB_DIR.mkdir(parents=True, exist_ok=True)
    act = conv_model(pre[:1], training=False).numpy()[0]            # (14, 14, 512)
    top_ch = np.argsort(act.max(axis=(0, 1)))[::-1][:5]
    neurons = []
    for ch in top_ch:
        r, c = np.unravel_index(np.argmax(act[..., ch]), act.shape[:2])
        neurons.append((int(ch), int(r), int(c)))

    fig, axes = plt.subplots(1, 6, figsize=(20, 3.8))
    axes[0].imshow(raw[0] / 255.0); axes[0].set_title(f"Input ('{class_names[0]}')", fontsize=9); axes[0].axis("off")
    for k, (ch, r, c) in enumerate(neurons, start=1):
        g = gb_to_rgb(guided_backprop(conv_model, pre[0], ch, r, c))
        plt.imsave(GB_DIR / f"neuron{k}_ch{ch}_r{r}_c{c}.png", g)
        axes[k].imshow(g); axes[k].set_title(f"Neuron {k}: ch {ch} @({r},{c})", fontsize=9); axes[k].axis("off")
    fig.suptitle("Guided backpropagation, block5_conv4 (14x14x512)", fontsize=11)
    fig.tight_layout(); fig.savefig(VGG_DIR / "guided_backprop_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (VGG_DIR / "guided_backprop_neurons.json").write_text(
        json.dumps([{"neuron": k, "channel": ch, "row": r, "col": c} for k, (ch, r, c) in enumerate(neurons, 1)], indent=2))


# ----------------------------------------------------------------------------
# Q2d: Grad-CAM (correct: weights = spatial mean of GRADIENTS)
# ----------------------------------------------------------------------------
def build_gradcam_models(model):
    last_conv = model.get_layer("block5_conv4")
    conv_model = keras.Model(model.input, last_conv.output)            # image -> block5_conv4 maps
    clf_in = keras.Input(shape=tuple(last_conv.output.shape[1:]))      # block5_conv4 maps -> logits
    x = clf_in
    for name in ["block5_pool", "flatten", "fc1", "fc2", "new_head"]:
        x = model.get_layer(name)(x)
    return conv_model, keras.Model(clf_in, x)


def grad_cam(conv_model, clf_model, x_pre, class_idx):
    with tf.GradientTape() as tape:
        conv_out = conv_model(x_pre, training=False)
        tape.watch(conv_out)                                           # frozen backbone => must watch explicitly
        score = clf_model(conv_out, training=False)[:, class_idx]      # pre-softmax logit
    grads = tape.gradient(score, conv_out)                             # d score / d feature maps
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)        # alpha_k
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    cam = tf.image.resize(cam[..., None], IMG_SIZE, method="bilinear")[..., 0].numpy()
    return cam, float(score[0])


def overlay(img01, heat, alpha=0.5):
    jet = matplotlib.colormaps["jet"](np.uint8(255 * np.clip(heat, 0, 1)))[..., :3]
    return np.clip(alpha * jet + (1 - alpha) * img01, 0, 1)


def run_gradcam(model, raw, pre, class_names):
    """5 images x 5 classes = 25 heatmaps. One figure per input image + individual overlay files."""
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    conv_model, clf_model = build_gradcam_models(model)
    probs = tf.nn.softmax(model(pre, training=False)).numpy()
    for i in range(5):
        fig, axes = plt.subplots(1, 6, figsize=(21, 3.9))
        axes[0].imshow(raw[i] / 255.0)
        axes[0].set_title(f"Input '{class_names[i]}'\npred: {class_names[int(probs[i].argmax())]} "
                          f"({probs[i].max():.2f})", fontsize=9)
        axes[0].axis("off")
        for j in range(5):
            cam, logit = grad_cam(conv_model, clf_model, pre[i:i + 1], j)
            ov = overlay(raw[i] / 255.0, cam)
            plt.imsave(GRADCAM_DIR / f"{class_names[i]}_explains_{class_names[j]}.png", ov)
            axes[j + 1].imshow(ov)
            tag = " (true)" if i == j else ""
            axes[j + 1].set_title(f"Target: {class_names[j]}{tag}\nlogit {logit:.2f}", fontsize=9)
            axes[j + 1].axis("off")
        fig.suptitle(f"Grad-CAM (block5_conv4) for the '{class_names[i]}' image, all 5 target classes", fontsize=11)
        fig.tight_layout()
        fig.savefig(GRADCAM_DIR / f"gradcam_{class_names[i]}_all_classes.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    keras.utils.set_random_seed(SEED)
    for d in (VGG_DIR, GRADCAM_DIR, GB_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_ds, class_names = make_ds("train", shuffle=True)
    train_eval_ds, _ = make_ds("train", shuffle=False)
    val_ds, _ = make_ds("val", shuffle=False)
    test_ds, _ = make_ds("test", shuffle=False)

    model = build_model()
    model.summary(line_length=100)
    trainable = [w.name for w in model.trainable_weights]
    print("Trainable weights (should only be new_head):", trainable)

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                                       restore_best_weights=True)])

    # --- Q2a: accuracy + confusion matrices -------------------------------------------------
    results = {}
    for split, ds in (("train", train_eval_ds), ("val", val_ds), ("test", test_ds)):
        acc, cm = evaluate(model, ds)
        results[split] = {"accuracy": acc, "confusion_matrix": cm.tolist()}
        plot_confusion_matrix(cm, class_names, f"VGG19 (head only) - {split} (Acc: {acc * 100:.1f}%)",
                              VGG_DIR / f"vgg19_{split}_confusion_matrix.png")
        print(f"[VGG19] {split}: {acc * 100:.2f}%")

    comp = {"vgg19_head_only": results}
    custom = RESULTS_DIR / "architecture_comparison_metrics.json"
    if custom.exists():                                   # written by part_b.py (Q1b)
        data = json.loads(custom.read_text())
        comp["custom_cnn_best"] = {"architecture": data["best_architecture"],
                                   **data["results"][data["best_architecture"]]}
    (VGG_DIR / "vgg19_vs_custom_comparison.json").write_text(json.dumps(comp, indent=2))

    # --- Q2b / Q2c / Q2d --------------------------------------------------------------------
    raw, pre, _ = load_representative_images("train")
    conv_model, _ = build_gradcam_models(model)
    plot_rf_patches(conv_model, raw, pre, class_names, VGG_DIR / "vgg19_rf_patches.png")
    run_guided_backprop(conv_model, raw, pre, class_names)
    run_gradcam(model, raw, pre, class_names)
    print(f"Saved everything to {VGG_DIR}")


if __name__ == "__main__":
    main()
```

### Sanity checks to run

1. `Trainable weights` printed in `main()` must list **only** `new_head/kernel` and `new_head/bias`.
2. In the Grad-CAM figure, the **heatmaps must differ between target classes** for the same image (with the old code they were identical).
3. Guided-backprop output should show sharp edge/texture structure, visibly crisper than plain-gradient saliency. As a check, set `GuidedReluMode.__enter__` to a no-op and compare. If they look the same, the activation swap isn't taking effect in your Keras version.
4. If your TensorFlow ≥ 2.16 (Keras 3) raises an error about `layer.activation` assignment, fall back to `tf_keras` (`pip install tf_keras` and `import tf_keras as keras`).

### B-5. Report note: VGG19 receptive field

The analytic RF of `block5_conv4` is **252×252 > 224×224**, so every bounding box is clipped to (almost) the whole image. This is mathematically correct, not a bug. State it in the report and explain that the *effective* RF is much smaller than the theoretical one. The guided-backprop images (Q2c) show where the neuron really looks.

---

# Repo / submission hygiene

## G-3. Cleanup and packaging

- `utils/select_images.py` is unused (both scripts re-implement sampling). Delete it or import it in both scripts. `plot_grid`, `plot_filters`, `plot_feature_maps`, `plot_bounding_boxes` in `utils/plotting.py` are also unused (`plot_filters` only draws input-channel 0 of each kernel).
- Folder layout required by the brief (code must be `.py`):

```
Group30_Assignment5_code/
├── part_a.py
├── part_b.py
├── part_b_vgg.py
└── utils/
    ├── __init__.py        (add an empty file to be safe)
    ├── metrics.py
    ├── plotting.py
    ├── receptive_field.py
    └── seed.py
```
  Zip as `Group30_Assignment5_code.zip`; report as `Group30_Assignment5_report.pdf`. The brief says submissions with the wrong naming will not be accepted.
- **Deadline:** the PDF says "Sunday, October 31, 2026", but 31 Oct 2026 is a **Saturday**. Confirm the real deadline with the TA.

## Things to state explicitly in the report

1. Conv1 has 6 filters, so "10 filters from the first layer" (Q3b) was met by plotting all 6.
2. The CNN uses mini-batch SGD with momentum 0.9 and weight decay (or vanilla SGD if you set `--momentum 0`), and a logits + `CrossEntropyLoss` formulation equals softmax output.
3. "Last convolutional layer" = conv output before pooling (Part A: 8×8, RF 14×14; Part B Arch1: 22×22, RF 51×51; VGG19: 14×14, RF 252×252).
4. Which seed was used, and the chosen filter indices (Q2 = 0,1,2; Q3b = 0–5 and 0–9).

## Verified as correct (no change needed)

- Part A architecture and shape trace (28→24→12→8→4; flatten 256; FC 64; FC 5); Kaiming (conv) / Xavier (FC) initialisation.
- `receptive_field.py` formula and `neuron_to_bounding_box` (Part A conv2 RF = 14, jump 2, boxes `[2r, 2r+14)`).
- Part B Arch1/2/3 definitions and the layer-spec construction for the RF calculation.
- Consistent class ordering between `ImageFolder` and the sorted-directory sampling; the same first-image-per-class is used in every task.
