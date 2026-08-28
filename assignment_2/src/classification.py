# classification.py
# Entry point for the classification pipeline (LS and NLS datasets) — FCNN version.
# Same filename/role as Assignment 1, but drives an FCNN sweep instead of OneVsOneClassifier.

# run_dataset(dataset_name, X, y, ...)
#   - Splits the data 60/20/20 (train/val/test), runs the architecture sweep via model_selection,
#     writes validation metrics for every architecture tried, evaluates the selected best model on
#     the test split, and generates its decision-region, error-vs-epoch, confusion-matrix, and
#     per-node output plots.
#
# main()
#   - Loads the LS and NLS datasets and calls run_dataset for each, producing all classification
#     results and plots for the report.
