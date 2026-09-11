# Execution Guide

All commands should be run from inside the `src/` directory.

---

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

### 2. Generate Initial Model Weights
Generates deterministic initial weights for the architectures so all optimizers start from identical weights:
```powershell
python models.py
```

---

### 3. Verify Data Loader
Loads images from `data/` and verifies dataset shapes and class mapping:
```powershell
python data_loader.py
```

---

### 4. Train Individual Model (Optional)
Trains a specific architecture and optimizer configuration (using ReLU):
```powershell
python train.py --arch arch_3l_v1 --optimizer adam
```

---

### 5. Run Full Comparative Experiments
Runs the experimentation suite across architectures and optimizers, superimposes error curves, generates summary comparison tables, and evaluates the best architecture on the test set:
```powershell
python compare_optimizers.py
```
