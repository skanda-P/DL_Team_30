# model_selection.py
# Orchestrates the architecture cross-validation sweep and selects the best model per dataset.
# New file — Assignment 1 had no architecture search (only a single fixed-size perceptron).

# architecture_grid(task_type)
#   - Generates the set of hidden-layer-count / hidden-node-count / activation combinations to try
#     for a given classification or regression dataset, per the assignment's requirement to
#     "try different numbers of hidden nodes" and compare architectures.
#
# run_sweep(X_train, y_train, X_val, y_val, task_type)
#   - Trains an FCNN for every architecture in the grid and evaluates each one on the validation
#     split, returning all trained models together with their validation metrics.
#
# select_best(results, metric_key, mode)
#   - Picks the best-performing architecture from the sweep results (mode="max" for accuracy or
#     F-measure, mode="min" for RMSE), which is then re-evaluated on the held-out test split.
