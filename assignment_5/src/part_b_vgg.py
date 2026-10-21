"""VGG19 Transfer Learning, Guided Backpropagation, and Grad-CAM Explanations.

This module implements the transfer learning and explainable AI pipeline using TensorFlow Keras:
1. VGG19 feature extraction backbone with ImageNet pretrained weights.
2. Fine-tuning the classification head for Caltech-101 5-class categorization.
3. Analytical receptive field calculation and distinct input patch localization for each class.
4. Guided Backpropagation with multiple visual modes (RGB gradients, saliency maps, and direct physical overlays).
5. Gradient-Weighted Class Activation Mapping (Grad-CAM) saved as individual images per class pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.metrics import plot_confusion_matrix
from utils.receptive_field import calculate_receptive_field, neuron_to_bounding_box
from utils.seed import set_seed

ROOT = Path(__file__).resolve().parent
CALTECH_DIR = ROOT / "data" / "caltech101_resized"
if not CALTECH_DIR.exists():
    CALTECH_DIR = ROOT / "data" / "caltech101_subset"

RESULTS_DIR = ROOT / "results" / "partB" / "vgg19"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5


def make_dataset(split: str, shuffle: bool):
    """Create a tf.data.Dataset for the given split with VGG19 preprocessing."""
    dataset = keras.utils.image_dataset_from_directory(
        CALTECH_DIR / split,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    class_names = dataset.class_names
    preprocess = keras.applications.vgg19.preprocess_input
    dataset = dataset.map(
        lambda images, labels: (preprocess(tf.cast(images, tf.float32)), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE), class_names


def load_representative_images():
    """Load one deterministic sample image per class directly from disk for reproducible analysis."""
    arrays = []
    class_names = []
    for class_dir in sorted(path for path in (CALTECH_DIR / "train").iterdir() if path.is_dir()):
        image_path = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]
        arrays.append(
            keras.utils.img_to_array(
                keras.utils.load_img(
                    image_path, target_size=IMG_SIZE, interpolation="bilinear"
                )
            )
        )
        class_names.append(class_dir.name)
    raw = np.asarray(arrays, dtype="float32")
    return raw, keras.applications.vgg19.preprocess_input(raw.copy()), class_names


def build_model() -> keras.Model:
    """Construct VGG19 with frozen convolutional backbone and retrainable 5-class head."""
    base = keras.applications.VGG19(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False
    logits = keras.layers.Dense(NUM_CLASSES, name="new_head")(base.layers[-2].output)
    model = keras.Model(base.input, logits, name="vgg19_5class")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def evaluate(model, dataset):
    """Compute overall accuracy and confusion matrix for a dataset."""
    labels, predictions = [], []
    for images, targets in dataset:
        labels.append(targets.numpy())
        predictions.append(np.argmax(model(images, training=False).numpy(), axis=1))
    labels, predictions = np.concatenate(labels), np.concatenate(predictions)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    return float(np.mean(labels == predictions)), matrix


def vgg_receptive_field():
    """Calculate the cumulative receptive field for VGG19 up to block5_conv4."""
    specs = []
    for block, convolutions in enumerate((2, 2, 4, 4, 4)):
        specs.extend([(3, 1, 1)] * convolutions)
        if block < 4:
            specs.append((2, 2, 0))
    return calculate_receptive_field(specs)


@tf.custom_gradient
def guided_relu(values):
    """Custom ReLU activation enforcing non-negative gradients for Guided Backpropagation."""
    output = tf.nn.relu(values)

    def gradient(upstream):
        return tf.cast(values > 0, upstream.dtype) * tf.cast(upstream > 0, upstream.dtype) * upstream

    return output, gradient


class GuidedReluMode:
    """Context manager replacing standard ReLU activations in the model with guided ReLU."""

    def __init__(self, model):
        self.model = model
        self.saved = {}

    def __enter__(self):
        for layer in self.model.layers:
            activation = getattr(layer, "activation", None)
            if activation is keras.activations.relu or getattr(activation, "__name__", "") == "relu":
                self.saved[layer] = activation
                layer.activation = guided_relu
        if not self.saved:
            raise RuntimeError(
                "Guided backpropagation could not find any ReLU activations to replace"
            )
        return self

    def __exit__(self, *_):
        for layer, activation in self.saved.items():
            layer.activation = activation


def run_explanations(model, raw, pre, class_names):
    """Run receptive field localization, guided backpropagation, and individual Grad-CAM heatmaps."""
    conv_layer = model.get_layer("block5_conv4")
    conv_model = keras.Model(model.input, conv_layer.output)
    rf, jump, start = vgg_receptive_field()
    activations = conv_model(pre, training=False).numpy()

    # ========================================================================
    # Distinct Receptive Field Patches (One per Class)
    # ========================================================================
    info = {}
    for index, class_name in enumerate(class_names):
        row, col, channel = np.unravel_index(np.argmax(activations[index]), activations[index].shape)
        bbox = neuron_to_bounding_box(row, col, rf, jump, start, (224, 224))
        ymin, xmin, ymax, xmax = bbox
        info[class_name] = {"channel": int(channel), "row": int(row), "col": int(col), "bbox": list(bbox)}

        image = raw[index] / 255.0

        # Save distinct 2-panel figure for this individual class: [Full image with box | Cropped patch]
        fig_patch, (ax_full, ax_crop) = plt.subplots(1, 2, figsize=(8.5, 4))
        ax_full.imshow(image)
        ax_full.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="cyan", linewidth=2.5))
        ax_full.set_title(f"Class: '{class_name}'\nMax Act: Ch {channel+1} ({row}, {col})", fontsize=10)
        ax_full.axis("off")

        ax_crop.imshow(image[ymin:ymax, xmin:xmax])
        ax_crop.set_title(f"Receptive Field Patch\n[{ymax-ymin}x{xmax-xmin} pixels, RF={int(rf)}x{int(rf)}]", fontsize=10)
        ax_crop.axis("off")

        fig_patch.suptitle(f"VGG19 Receptive Field Patch Localization: Class '{class_name}'", fontsize=11)
        fig_patch.tight_layout()
        distinct_patch_path = RESULTS_DIR / f"vgg19_rf_patch_{class_name}.png"
        fig_patch.savefig(distinct_patch_path, dpi=180, bbox_inches="tight")
        plt.close(fig_patch)
        print(f"Saved distinct VGG19 receptive field patch for '{class_name}' to {distinct_patch_path}")

    (RESULTS_DIR / "vgg19_rf_patch_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # ========================================================================
    # ========================================================================
    # Guided Backpropagation (4-Panel Representation for ALL 5 Representative Images)
    # ========================================================================
    guided_info = []

    for index, class_name in enumerate(class_names):
        row, col, channel = np.unravel_index(np.argmax(activations[index]), activations[index].shape)
        act_val = float(activations[index, row, col, channel])
        guided_info.append({
            "class": class_name,
            "channel": int(channel),
            "row": int(row),
            "col": int(col),
            "activation": act_val,
        })
        raw_norm = raw[index] / 255.0
        image = tf.convert_to_tensor(pre[index:index+1])
        with GuidedReluMode(conv_model):
            with tf.GradientTape() as tape:
                tape.watch(image)
                score = conv_model(image, training=False)[0, row, col, channel]
            gradient = tape.gradient(score, image)[0].numpy()

        # 1. Normalized RGB gradient
        norm_rgb = (gradient - gradient.mean()) / (gradient.std() + 1e-8) * 0.15 + 0.5
        norm_rgb = np.clip(norm_rgb, 0.0, 1.0)

        # 2. Maximum magnitude saliency map across color channels
        saliency = np.max(np.abs(gradient), axis=-1)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # 3. Direct physical overlay: Blend inferno saliency directly onto dimmed input image
        inferno_rgb = plt.get_cmap("inferno")(saliency)[:, :, :3]
        dimmed_orig = 0.35 * raw_norm
        overlay = np.clip(dimmed_orig + 0.65 * inferno_rgb, 0.0, 1.0)

        # Save standalone individual full-resolution images for each class
        plt.imsave(RESULTS_DIR / f"guided_backprop_{class_name}_saliency.png", saliency, cmap="inferno")
        plt.imsave(RESULTS_DIR / f"guided_backprop_{class_name}_rgb.png", norm_rgb)
        plt.imsave(RESULTS_DIR / f"guided_backprop_{class_name}_overlay.png", overlay)

        # Save distinct 4-panel figure for this representative class
        fig_gb, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
        ax1.imshow(raw_norm)
        ax1.set_title(f"Input: '{class_name}'", fontsize=9)
        ax1.axis("off")

        ax2.imshow(norm_rgb)
        ax2.set_title(f"RGB Gradient", fontsize=9)
        ax2.axis("off")

        ax3.imshow(saliency, cmap="inferno")
        ax3.set_title(f"Saliency Map", fontsize=9)
        ax3.axis("off")

        ax4.imshow(overlay)
        ax4.set_title(f"Saliency Overlay", fontsize=9)
        ax4.axis("off")

        fig_gb.suptitle(f"Guided Backpropagation Attribution: Class '{class_name}' (Channel {channel+1} at [{row}, {col}])", fontsize=11)
        fig_gb.tight_layout()
        distinct_gb_path = RESULTS_DIR / f"guided_backprop_{class_name}.png"
        fig_gb.savefig(distinct_gb_path, dpi=180, bbox_inches="tight")
        plt.close(fig_gb)
        print(f"Saved distinct Guided Backprop card for {class_name} to {distinct_gb_path}")

    # Also generate the top-5 neuron breakdowns for the first representative image ('butterfly')
    first_activation = activations[0]
    channels = np.argsort(first_activation.max(axis=(0, 1)))[::-1][:5]
    raw0_norm = raw[0] / 255.0
    for output_index, channel in enumerate(channels, start=1):
        row, col = np.unravel_index(np.argmax(first_activation[..., channel]), first_activation.shape[:2])
        image = tf.convert_to_tensor(pre[0:1])
        with GuidedReluMode(conv_model):
            with tf.GradientTape() as tape:
                tape.watch(image)
                score = conv_model(image, training=False)[0, row, col, channel]
            gradient = tape.gradient(score, image)[0].numpy()

        norm_rgb = (gradient - gradient.mean()) / (gradient.std() + 1e-8) * 0.15 + 0.5
        norm_rgb = np.clip(norm_rgb, 0.0, 1.0)
        saliency = np.max(np.abs(gradient), axis=-1)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        inferno_rgb = plt.get_cmap("inferno")(saliency)[:, :, :3]
        overlay = np.clip(0.35 * raw0_norm + 0.65 * inferno_rgb, 0.0, 1.0)

        plt.imsave(RESULTS_DIR / f"guided_backprop_ch{channel}_saliency.png", saliency, cmap="inferno")
        plt.imsave(RESULTS_DIR / f"guided_backprop_ch{channel}_rgb.png", norm_rgb)
        plt.imsave(RESULTS_DIR / f"guided_backprop_ch{channel}_overlay.png", overlay)

        fig_gb, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
        ax1.imshow(raw0_norm); ax1.set_title(f"Input: '{class_names[0]}'", fontsize=9); ax1.axis("off")
        ax2.imshow(norm_rgb); ax2.set_title("RGB Gradient", fontsize=9); ax2.axis("off")
        ax3.imshow(saliency, cmap="inferno"); ax3.set_title("Saliency Map", fontsize=9); ax3.axis("off")
        ax4.imshow(overlay); ax4.set_title("Saliency Overlay", fontsize=9); ax4.axis("off")
        fig_gb.suptitle(f"Guided Backpropagation Attribution: Neuron {output_index} (Channel {channel+1} at [{row}, {col}])", fontsize=11)
        fig_gb.tight_layout()
        distinct_gb_path = RESULTS_DIR / f"guided_backprop_neuron_{output_index}_ch_{channel}.png"
        fig_gb.savefig(distinct_gb_path, dpi=180, bbox_inches="tight")
        plt.close(fig_gb)

    (RESULTS_DIR / "guided_backprop_info.json").write_text(
        json.dumps(guided_info, indent=2), encoding="utf-8"
    )

    # ========================================================================
    # Grad-CAM Localization (Saved Strictly as Individual Images per Class Pair)
    # ========================================================================
    gradcam_dir = RESULTS_DIR / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    classifier = keras.Model(conv_layer.output, model.output)

    for image_index, image in enumerate(pre):
        input_class = class_names[image_index]
        orig_img_norm = raw[image_index] / 255.0
        convolution = conv_model(image[None], training=False)

        for class_index in range(NUM_CLASSES):
            target_class = class_names[class_index]
            with tf.GradientTape() as tape:
                tape.watch(convolution)
                score = classifier(convolution, training=False)[:, class_index]
            gradients = tape.gradient(score, convolution)
            weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
            heatmap = tf.nn.relu(tf.reduce_sum(weights * convolution, axis=-1))[0].numpy()
            heatmap = heatmap / (heatmap.max() + 1e-8)
            heatmap = tf.image.resize(heatmap[..., None], IMG_SIZE)[..., 0].numpy()

            # Colorize heatmap with jet colormap and overlay on original image
            color_heat = plt.get_cmap("jet")(heatmap)[:, :, :3]
            overlay = np.clip(0.5 * color_heat + 0.5 * orig_img_norm, 0.0, 1.0)

            # Save standalone overlay
            plt.imsave(gradcam_dir / f"{input_class}_explains_{target_class}_overlay.png", overlay)

            # Save individual 3-panel figure: [Input Image | Pure Heatmap | Overlaid Visualization]
            fig_cam, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(orig_img_norm)
            ax1.set_title(f"Input: {input_class}", fontsize=10)
            ax1.axis("off")

            ax2.imshow(heatmap, cmap="jet")
            pred_score = float(score[0]) if score.shape[0] > 0 else 0.0
            ax2.set_title(f"Grad-CAM Heatmap (Score: {pred_score:.3f})", fontsize=10)
            ax2.axis("off")

            ax3.imshow(overlay)
            match_flag = "[GROUND TRUTH]" if input_class == target_class else "[COUNTERFACTUAL]"
            ax3.set_title(f"Overlaid: {target_class} {match_flag}", fontsize=10, color="#007700" if input_class == target_class else "#222222")
            ax3.axis("off")

            fig_cam.suptitle(f"Grad-CAM Localization: Explaining Prediction for '{target_class}' on '{input_class}' Image", fontsize=11)
            fig_cam.tight_layout()

            individual_cam_path = gradcam_dir / f"{input_class}_explains_{target_class}.png"
            fig_cam.savefig(individual_cam_path, dpi=180, bbox_inches="tight")
            plt.close(fig_cam)

    print(f"Saved 25 individual Grad-CAM heatmaps and overlays to {gradcam_dir}")


def main():
    parser = argparse.ArgumentParser(description="VGG19 Transfer Learning and Visual Interpretability Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs for VGG19 classification head training (default: 30)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (default: 8)")
    args = parser.parse_args()
    set_seed(42)
    keras.utils.set_random_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gradcam").mkdir(exist_ok=True)

    train, class_names = make_dataset("train", True)
    train_eval, _ = make_dataset("train", False)
    val, _ = make_dataset("val", False)
    test, _ = make_dataset("test", False)

    model = build_model()
    print("Trainable weights:", [weight.name for weight in model.trainable_weights])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    history = model.fit(
        train,
        validation_data=val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Plot VGG19 training curves
    epochs_trained = len(history.history["loss"])
    epochs_range = range(1, epochs_trained + 1)
    fig_curves, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_l.plot(epochs_range, history.history["loss"], label="Train Loss", color="#2563EB", lw=2)
    ax_l.plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#DC2626", lw=2, linestyle="--")
    ax_l.set_title("VGG19 Loss Convergence", fontsize=10, fontweight="bold")
    ax_l.set_xlabel("Epoch", fontsize=9)
    ax_l.set_ylabel("Categorical Cross-Entropy", fontsize=9)
    ax_l.grid(True, linestyle="--", alpha=0.3)
    ax_l.legend(fontsize=8)

    ax_a.plot(epochs_range, [acc * 100 for acc in history.history["accuracy"]], label="Train Accuracy", color="#2563EB", lw=2)
    ax_a.plot(epochs_range, [acc * 100 for acc in history.history["val_accuracy"]], label="Val Accuracy", color="#059669", lw=2, linestyle="--")
    ax_a.set_title("VGG19 Accuracy Progression", fontsize=10, fontweight="bold")
    ax_a.set_xlabel("Epoch", fontsize=9)
    ax_a.set_ylabel("Accuracy (%)", fontsize=9)
    ax_a.grid(True, linestyle="--", alpha=0.3)
    ax_a.legend(fontsize=8)

    fig_curves.suptitle("VGG19 Transfer Learning Optimization Dynamics", fontsize=11, fontweight="bold")
    fig_curves.tight_layout()
    fig_curves.savefig(RESULTS_DIR / "vgg19_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_curves)
    print(f"Saved VGG19 training curves to {RESULTS_DIR / 'vgg19_training_curves.png'}")

    # Evaluate classification metrics across dataset splits
    metrics = {}
    for split, dataset in (("train", train_eval), ("val", val), ("test", test)):
        value, matrix = evaluate(model, dataset)
        metrics[split] = {"accuracy": value, "confusion_matrix": matrix.tolist()}
        plot_confusion_matrix(
            matrix,
            class_names,
            f"VGG19 Transfer Learning - {split.capitalize()} Set ({value * 100:.1f}%)",
            RESULTS_DIR / f"vgg19_{split}_confusion_matrix.png",
        )
    (RESULTS_DIR / "vgg19_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Comparative evaluation with custom CNN architecture
    custom_metrics_path = RESULTS_DIR.parent / "architecture_comparison_metrics.json"
    if custom_metrics_path.exists():
        custom = json.loads(custom_metrics_path.read_text(encoding="utf-8"))
        best_architecture = custom.get("best_architecture", "arch1")
        custom_result = custom.get("results", {}).get(best_architecture, {})
        if custom_result:
            comparison = {
                "custom_cnn": {
                    "architecture": best_architecture,
                    "test_accuracy": custom_result.get("test_acc", 0.0),
                    "test_confusion_matrix": custom_result.get("test_cm", []),
                },
                "vgg19": {
                    "test_accuracy": metrics["test"]["accuracy"],
                    "test_confusion_matrix": metrics["test"]["confusion_matrix"],
                },
            }
            (RESULTS_DIR / "custom_cnn_vs_vgg19_test_comparison.json").write_text(
                json.dumps(comparison, indent=2), encoding="utf-8"
            )
            (RESULTS_DIR / "q1_q2_test_comparison.json").write_text(
                json.dumps(comparison, indent=2), encoding="utf-8"
            )
            figure, axes = plt.subplots(1, 2, figsize=(11, 5))
            for axis, title, matrix, value in (
                (axes[0], f"Best Custom CNN: {best_architecture.upper()} ({custom_result.get('test_acc', 0.0) * 100:.1f}%)",
                 np.asarray(custom_result.get("test_cm", np.zeros((NUM_CLASSES, NUM_CLASSES)))), custom_result.get("test_acc", 0.0)),
                (axes[1], f"VGG19 Transfer Learning ({metrics['test']['accuracy'] * 100:.1f}%)",
                 np.asarray(metrics["test"]["confusion_matrix"]), metrics["test"]["accuracy"]),
            ):
                axis.imshow(matrix, cmap="Blues")
                axis.set_title(title, fontsize=10, fontweight="bold")
                axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
                axis.set_yticks(range(len(class_names)), class_names)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("True")
                threshold = matrix.max() / 2 if matrix.max() else 1
                for row in range(NUM_CLASSES):
                    for column in range(NUM_CLASSES):
                        axis.text(
                            column, row, str(matrix[row, column]),
                            ha="center", va="center",
                            color="white" if matrix[row, column] > threshold else "black",
                            fontweight="bold",
                        )
            figure.suptitle("Comparative Test Set Performance: Best Custom CNN vs. Pretrained VGG19", fontsize=12, fontweight="bold")
            figure.tight_layout()
            figure.savefig(RESULTS_DIR / "custom_cnn_vs_vgg19_test_comparison.png", dpi=180, bbox_inches="tight")
            figure.savefig(RESULTS_DIR / "q1_q2_test_comparison.png", dpi=180, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved comparative test confusion matrix to {RESULTS_DIR / 'custom_cnn_vs_vgg19_test_comparison.png'}")
    else:
        print(f"Notice: {custom_metrics_path} not found. Run part_b.py first to generate the custom vs VGG19 comparison figure.")

    raw, pre, representative_names = load_representative_images()
    run_explanations(model, raw, pre, representative_names)


if __name__ == "__main__":
    main()
