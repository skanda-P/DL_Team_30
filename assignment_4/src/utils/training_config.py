# Hyperparameters, training configurations (epochs, learning rates, batch sizes, bottleneck sizes), and architecture definitions.


SEED = 42

# ---- Convergence criterion (same as Assignment 3) ----
MAX_EPOCHS = 10000
STOPPING_THRESHOLD = 1e-4
PATIENCE = 1

# ---- Classifier optimiser (best optimiser from Assignment 3: SGD + momentum) ----
BATCH_SIZE = 1
LEARNING_RATE = 0.001
MOMENTUM = 0.9
ACTIVATION = "tanh"

# ---- Task 1 ----
PCA_DIMS = [32, 64, 128, 256]

# ---- Best result of Assignment 3 (Group30_Assignment3_report.pdf) ----
# Used for the "compare with Assignment 3" step of Tasks 1, 3 and 4.
A3_REFERENCE = {
    "architecture": "arch_4l_v3",
    "hidden_dims": [384, 256, 128, 64],
    "optimizer": "SGD + Momentum (batch_size=1, lr=0.001, momentum=0.9)",
    "input_dim": 784,
    "val_acc": 98.39,
    "test_acc": 98.02,
    "test_macro_f1": 98.02,
}