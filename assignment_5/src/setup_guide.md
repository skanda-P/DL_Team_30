# Assignment 5 Setup and Execution Guide

## 1. Directory Structure

```
assignment_5/
├── CS601T_Assignment5_Details_12October2026.pdf
└── src/
    ├── data/
    │   ├── mnist_subset/            <- MNIST 5 classes (3, 4, 5, 7, 8)
    │   ├── caltech101_subset/       <- Raw Caltech-101 images
    │   └── caltech101_resized/      <- Pre-resized 224x224 RGB images
    ├── models/                      <- Checkpoints (mnist_cnn.pth, caltech_best.pth)
    ├── results/
    │   ├── partA/                   <- Figures and JSON metrics for MNIST experiments
    │   │   ├── filters/             <- Dedicated folder with each filter saved individually with cell values
    │   │   │   ├── single_filter_kernel.png
    │   │   │   ├── stacked_conv1_filter_{1..3}.png & stacked_conv2_filter_{1..3}.png
    │   │   │   ├── trained_conv1_filter_{1..6}.png
    │   │   │   └── trained_conv2_filter_{1..16}.png
    │   │   ├── feature_maps/        <- Atomic feature maps per class and compact report-friendly grids
    │   │   │   ├── class_{class}/   <- Individual atomic feature maps (Conv1, Conv2, Pool, RF patches)
    │   │   │   ├── stacked_features_{class}_conv{1,2}_relu.png (clean 1x3 strips)
    │   │   │   └── trained_features_{class}_conv{1,2}.png (clean 2x3 and 2x5 grids)
    │   │   ├── single_filter_{class}.png (distinct 3-panel per class with annotated filter values)
    │   │   ├── receptive_field_patch_{class}.png (distinct per class with bounding boxes)
    │   │   └── mnist_training_curves.png & mnist_confusion_matrices.png
    │   └── partB/                   <- Figures and JSON metrics for Caltech-101
    │       ├── dropout_study_grouped_bar_chart.png
    │       ├── q1_q2_test_comparison.png & q1_q2_test_comparison.json
    │       ├── custom_cnn/          <- Custom CNN benchmarks with Adam optimizer (lr=0.001)
    │       │   ├── dropout_study_summary.csv & overall_best.json
    │       │   └── dropout{0,0.3,0.5}/
    │       │       ├── arch{1,2,3}_architecture.md (layer table with output shapes and parameter counts)
    │       │       ├── arch{1,2,3}_{train,val,test}_confusion_matrix.png (separate atomic images)
    │       │       ├── arch{1,2,3}_training_curves.png (loss & validation accuracy curves)
    │       │       ├── arch1_feature_maps.png (post-ReLU feature maps for butterfly)
    │       │       ├── custom_cnn_rf_patch_{class}.png (distinct 2-panel per class with bounding boxes)
    │       │       ├── custom_cnn_patch_info.json
    │       │       └── architecture_comparison_metrics.json
    │       └── vgg19/               <- TensorFlow Keras VGG-19 transfer learning, Guided Backprop, and Grad-CAM
    │           ├── vgg19_{train,val,test}_confusion_matrix.png & vgg19_metrics.json
    │           ├── vgg19_training_curves.png
    │           ├── vgg19_rf_patch_{class}.png & vgg19_rf_patch_info.json
    │           ├── guided_backprop_neuron_{1..5}_ch_{ch}.png (distinct 4-panel per neuron)
    │           ├── guided_backprop_ch{ch}_{overlay,saliency,rgb}.png (standalone images)
    │           ├── guided_backprop_neuron_info.json
    │           └── gradcam/         <- 25 individual Grad-CAM heatmaps and overlays
    │               ├── {input}_explains_{target}.png (distinct 3-panel per pair)
    │               └── {input}_explains_{target}_overlay.png (standalone overlay)
    ├── utils/                       <- Utility modules
    │   ├── metrics.py               <- Accuracy and confusion matrix computation
    │   ├── plotting.py              <- Visual plotting routines (annotated filter values, distinct plots)
    │   ├── receptive_field.py       <- Analytical receptive field and bounding box calculation
    │   ├── seed.py                  <- Reproducible random seed utility
    │   └── select_images.py         <- Image selection helper
    ├── part_a.py                    <- MNIST convolutional pipeline
    ├── part_b.py                    <- Caltech-101 custom CNN architecture pipeline
    ├── part_b_vgg.py                <- TensorFlow Keras VGG19 and XAI pipeline
    ├── requirements.txt             <- Dependencies
    └── setup_guide.md
```

---

## 2. Dataset Setup & Folder Placement (Required for Reproduction)

Because large image datasets and model weights are excluded from Git tracking via `.gitignore` (to keep the GitHub repository fast and under size limits), anyone cloning this repository must place the datasets inside `src/data/` as instructed below before running the code.

> **Note on GitHub Tracking**: All experimental output figures, metrics, confusion matrices, and visualizations in `src/results/` **are committed and pushed to GitHub**. Only the raw images in `src/data/` and weights in `src/models/` must be placed locally.

### A. Part A: MNIST 5-Class Subset (`src/data/mnist_subset/`)
- **Assigned Classes (Group 30)**: `3`, `4`, `5`, `7`, `8` (same digit subset from Assignment 3 & 4).
- **Directory Layout**: PyTorch `torchvision.datasets.ImageFolder` expects the dataset to be organized into three split directories (`train`, `val`, `test`), each containing subdirectories named after the respective class label:

```
src/data/mnist_subset/
├── train/
│   ├── 3/          <- 2,277 grayscale images (28x28)
│   ├── 4/          <- 2,277 grayscale images (28x28)
│   ├── 5/          <- 2,277 grayscale images (28x28)
│   ├── 7/          <- 2,277 grayscale images (28x28)
│   └── 8/          <- 2,277 grayscale images (28x28)
├── val/
│   ├── 3/          <- 759 grayscale images (28x28)
│   ├── 4/          <- 759 grayscale images (28x28)
│   ├── 5/          <- 759 grayscale images (28x28)
│   ├── 7/          <- 759 grayscale images (28x28)
│   └── 8/          <- 759 grayscale images (28x28)
└── test/
    ├── 3/          <- 759 grayscale images (28x28)
    ├── 4/          <- 759 grayscale images (28x28)
    ├── 5/          <- 759 grayscale images (28x28)
    ├── 7/          <- 759 grayscale images (28x28)
    └── 8/          <- 759 grayscale images (28x28)
```
- **Total Images**: 18,975 images across all three splits (60% train / 20% validation / 20% test).
- **Supported Formats**: `.png`, `.jpg`, `.jpeg`. Images are converted to single-channel tensor grayscale (`1 x 28 x 28`) automatically.

---

### B. Part B: Caltech-101 5-Class Subset (`src/data/caltech101_subset/`)
- **Assigned Classes (Group 30)**: `butterfly`, `car_side`, `grand_piano`, `scorpion`, `starfish`.
- **Directory Layout**: Both PyTorch (`part_b.py`) and TensorFlow Keras (`part_b_vgg.py`) expect the standard `ImageFolder` / `image_dataset_from_directory` format:

```
src/data/caltech101_subset/
├── train/
│   ├── butterfly/    <- 50 raw RGB images (varying resolutions)
│   ├── car_side/     <- 50 raw RGB images (varying resolutions)
│   ├── grand_piano/  <- 50 raw RGB images (varying resolutions)
│   ├── scorpion/     <- 50 raw RGB images (varying resolutions)
│   └── starfish/     <- 50 raw RGB images (varying resolutions)
├── val/
│   ├── butterfly/    <- 10 raw RGB images (varying resolutions)
│   ├── car_side/     <- 10 raw RGB images (varying resolutions)
│   ├── grand_piano/  <- 10 raw RGB images (varying resolutions)
│   ├── scorpion/     <- 10 raw RGB images (varying resolutions)
│   └── starfish/     <- 10 raw RGB images (varying resolutions)
└── test/
    ├── butterfly/    <- 20 raw RGB images (varying resolutions)
    ├── car_side/     <- 20 raw RGB images (varying resolutions)
    ├── grand_piano/  <- 20 raw RGB images (varying resolutions)
    ├── scorpion/     <- 20 raw RGB images (varying resolutions)
    └── starfish/     <- 20 raw RGB images (varying resolutions)
```
- **Total Images**: 400 images (250 train / 50 validation / 100 test).
- **Raw Images vs Pre-Resized Cache (`caltech101_resized/`)**:
  - **Option 1 (Default / Simplest)**: Simply place your raw images in `src/data/caltech101_subset/`. Both `part_b.py` and `part_b_vgg.py` automatically detect this folder and resize all images to `224x224` dynamically during batch loading.
  - **Option 2 (Pre-resized Cache for Fast Training)**: To skip on-the-fly resizing during training, run:
    ```powershell
    python verify_dataset.py --resize_caltech
    ```
    This automatically generates `src/data/caltech101_resized/` with all images saved at `224x224` resolution.

---

### C. Automated Dataset Verification
To verify that all folders and classes are correctly placed and formatted, run:
```powershell
python verify_dataset.py
```
This script validates split existence, class folder names, image file extensions, and reports exact image counts.

---

## 3. Environment Setup

```powershell
pip install -r requirements.txt
```

---

## 4. Running the Code

### MNIST Convolutional Neural Network Pipeline (`part_a.py`)
Run the complete pipeline:
```powershell
python part_a.py
```
Or run individual stages via the `--pipeline` flag:
```powershell
python part_a.py --pipeline single_filter
python part_a.py --pipeline stacked_layers
python part_a.py --pipeline classifier --epochs 15
```

Generated outputs in `results/partA/`:
- `filters/`: Dedicated folder with each filter saved individually with cell weight values
  - `single_filter_kernel.png`: Single 5x5 filter with exact cell weight values
  - `stacked_conv1_filter_{1..3}.png`: Individual Conv1 filters with cell weight values
  - `stacked_conv2_filter_{1..3}.png`: Individual Conv2 filters with cell weight values
  - `trained_conv1_filter_{1..6}.png`: All 6 trained Conv1 filters saved individually with cell values
  - `trained_conv2_filter_{1..16}.png`: All 16 trained Conv2 filters saved individually with cell values
- `feature_maps/`: Atomic feature maps per class and compact report-friendly grids
  - `class_{class}/`: Atomic standalone images (`trained_conv1_f{1..6}.png`, `trained_conv2_f{1..16}.png`, `rf_patch_rank{1..5}_ch{ch}.png`, etc.)
  - `stacked_features_{class}_conv{1,2}_relu.png`: Clean 1x3 strips for quick comparison
  - `trained_features_{class}_conv1.png`: Clean 2x3 grid (6 filters)
  - `trained_features_{class}_conv2.png`: Clean 2x5 grid (10 filters)
- `single_filter_{class}.png`: Distinct 3-panel plot per class (Input | 5x5 Filter Heatmap with Cell Values | Feature Map)
- `single_filter_observations.json`
- `stacked_feature_maps.npz` & `stacked_convolution_observations.json`
- `mnist_training_curves.png` & `mnist_confusion_matrices.png` & `mnist_metrics.json`
- `receptive_field_patch_{class}.png`: Distinct receptive field patch plot for each class
- `receptive_field_patch_info.json`
- `models/mnist_cnn.pth`

---

### Caltech-101 Custom CNN Architecture Benchmark (`part_b.py`)
Run all three custom CNN architectures using the Adam optimizer with dynamic convergence / Early Stopping:
```powershell
python part_b.py --max_epochs 100 --patience 15 --lr 0.001
```

Generated outputs in `results/partB/custom_cnn/`:
- `arch{1,2,3}_architecture.md`: Model architecture layer tables detailing exact output shapes and parameter counts
- `arch{1,2,3}_train_confusion_matrix.png`: Separate individual train confusion matrix
- `arch{1,2,3}_val_confusion_matrix.png`: Separate individual validation confusion matrix
- `arch{1,2,3}_test_confusion_matrix.png`: Separate individual test confusion matrix
- `arch{1,2,3}_training_curves.png`: Loss and validation accuracy trajectories with best checkpoint marker
- `{best_arch}_feature_maps.png`: Layerwise feature maps for representative image (8 from Conv1 and 8 from remaining conv layers)
- `custom_cnn_rf_patch_{class}.png`: Distinct 2-panel RF patch localization for each class with bounding boxes
- `custom_cnn_patch_info.json`: Analytical receptive field and bounding box coordinates for each class
- `custom_cnn_architectures_summary.png`: Consolidated master publication figure comparing all 3 architectures under Adam (grouped train/val/test accuracy bar chart and detailed empirical results table)
- `custom_cnn_architectures_summary.json`: Complete numerical record of all 3 architectures (learning rates, epochs to converge, train/val/test accuracies, and confusion matrices)
- `architecture_comparison_metrics.json`: Top-performing model metrics across architectures for seamless VGG19 comparative analysis
- `models/caltech_best.pth`: Best overall model weights (ARCH1)
- `models/caltech_{arch1,arch2,arch3}.pth`: Saved weights for each individual architecture

---

### VGG19 Transfer Learning & Interpretability (`part_b_vgg.py`)
Run VGG19 transfer learning with TensorFlow Keras and Early Stopping:
```powershell
python part_b_vgg.py --epochs 30 --patience 8
```

Generated outputs in `results/partB/vgg19/`:
- `vgg19_{train,val,test}_confusion_matrix.png`: Separate individual confusion matrices for each split
- `vgg19_training_curves.png`: Training & validation loss and accuracy progression
- `vgg19_metrics.json`
- `custom_cnn_vs_vgg19_test_comparison.png` & `custom_cnn_vs_vgg19_test_comparison.json`: Best Custom CNN vs VGG19 comparison
- `vgg19_rf_patch_{class}.png`: Distinct 2-panel receptive field patch plot for each class
- `vgg19_rf_patch_info.json`
- `guided_backprop_neuron_{idx}_ch_{ch}.png`: Distinct 4-panel card for each neuron (Input | RGB Gradient | Saliency | Physical Overlay)
- `guided_backprop_ch{ch}_overlay.png`, `guided_backprop_ch{ch}_saliency.png`, `guided_backprop_ch{ch}_rgb.png`: Standalone full-resolution images
- `guided_backprop_info.json`
- `gradcam/`:
  - 25 distinct individual 3-panel figures: `{input_class}_explains_{target_class}.png`
  - 25 standalone individual overlay images: `{input_class}_explains_{target_class}_overlay.png`
  - (No composite heatmap grids generated)

---

## 5. Technical Notes

- **Filter Visualizations**: All filter heatmaps annotate the actual numerical weights on each cell with high-contrast text for direct legibility.
- **Image Resizing**: Both `part_b.py` and `part_b_vgg.py` source from `data/caltech101_resized` (pre-resized 224x224 RGB images) with fallback to `caltech101_subset`.
- **Softmax & Logits**: The CNN classifiers output raw logits and use `CrossEntropyLoss` / `SparseCategoricalCrossentropy(from_logits=True)`. Softmax is applied implicitly by the loss function for numerical stability.
- **Reproducibility**: The fixed random seed is set to `42` across Python, NumPy, PyTorch, and TensorFlow.
- **Receptive Field**: The reported "last convolutional layer" is the convolution output before pooling: 14x14 for MNIST Part A, 51x51 for Caltech Part B custom architectures, and 252x252 for VGG19 (clipped to 224x224 input).

---

## 6. Submission Packaging

- Code directory: `Group30_Assignment5_code/`
- Zip archive: `Group30_Assignment5_code.zip`
- Final Report: `Group30_Assignment5_report.pdf`
