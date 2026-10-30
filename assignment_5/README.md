# Deep Learning - Assignment 5: Convolutional Neural Networks & Interpretability

**Course**: CS601T - Deep Learning  
**Team**: Group 30  
**Tasks**:
- **Part A**: Convolutional layers, feature hierarchies, receptive fields, and supervised CNN classification on MNIST subset.
- **Part B**: Custom CNN architecture search (3 architectures x 3 SGD variants = 9 benchmark experiments) on Caltech-101 subset.
- **Part B (VGG19 & Interpretability)**: VGG19 Transfer Learning, Receptive Field localization, Guided Backpropagation (RGB gradients, saliency maps, physical overlays), and Grad-CAM 5x5 pair explanations.

---

## Results Status on GitHub

- **All Results are Pushed to GitHub**: All 190+ experiment figures, feature map cards, filter heatmaps with numerical cell values, receptive field bounding boxes, guided backprop cards, Grad-CAM overlays, and metric JSON files are tracked and available in `src/results/partA/` and `src/results/partB/`.
- **Datasets & Checkpoints Excluded**: Raw image data (`src/data/*`) and model checkpoints (`src/models/*.pth`) are excluded by `.gitignore` to maintain a lightweight repository.

---

## Dataset Setup Instructions (Required for Reproduction)

Before executing any training scripts, place your assigned datasets in `src/data/`:

### 1. MNIST Subset (Part A)
- **Path**: `src/data/mnist_subset/`
- **Assigned Classes (Group 30)**: `3`, `4`, `5`, `7`, `8`
- **Format**: Standard `ImageFolder` structure with `train/`, `val/`, `test/` splits:
  ```
  src/data/mnist_subset/
  ├── train/{3, 4, 5, 7, 8}/
  ├── val/{3, 4, 5, 7, 8}/
  └── test/{3, 4, 5, 7, 8}/
  ```

### 2. Caltech-101 Subset (Part B & VGG19)
- **Path**: `src/data/caltech101_subset/`
- **Assigned Classes (Group 30)**: `butterfly`, `car_side`, `grand_piano`, `scorpion`, `starfish`
- **Format**: Standard `ImageFolder` structure with `train/`, `val/`, `test/` splits:
  ```
  src/data/caltech101_subset/
  ├── train/{butterfly, car_side, grand_piano, scorpion, starfish}/
  ├── val/{butterfly, car_side, grand_piano, scorpion, starfish}/
  └── test/{butterfly, car_side, grand_piano, scorpion, starfish}/
  ```
- **Automatic Fallback**: The scripts automatically read raw images from `caltech101_subset/` and resize to 224x224 during execution.

### 3. Verify Dataset Placement
Run the verification utility to validate your folder structure:
```powershell
cd src
python verify_dataset.py
```
*(Optional: Pre-generate a 224x224 image cache by running `python verify_dataset.py --resize_caltech`)*

---

## Execution Guide

For full details, see [src/setup_guide.md](src/setup_guide.md).

```powershell
cd src
pip install -r requirements.txt

# Run MNIST pipeline (Part A)
python part_a.py

# Run Caltech-101 Custom CNN Benchmarks (Part B - 9 experiments)
python part_b.py --config all --max_epochs 100 --patience 15

# Run VGG19 Transfer Learning & Explainable AI (Part B - VGG19)
python part_b_vgg.py --epochs 30 --patience 8
```
