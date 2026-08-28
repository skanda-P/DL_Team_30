# regression.py
# Entry point for the regression pipeline (univariate and bivariate datasets) — FCNN version.
# Same filename/role as Assignment 1, but drives an FCNN sweep instead of a single perceptron.

# report_rmse(y_true, y_pred)
#   - Computes RMSE and %RMSE for a given prediction set. Carried over unchanged from Assignment 1.
#
# run_dataset(dataset_name, path, dim, ...)
#   - Splits the data 60/20/20, sweeps 1-hidden-layer (and additionally 2-hidden-layer for the
#     bivariate dataset) architectures, selects the best by validation RMSE, and generates
#     error-vs-epoch, target-vs-model, scatter, and per-node output plots across train/val/test
#     splits for the winning architecture.
#
# main()
#   - Loads the univariate and bivariate datasets and calls run_dataset for each.
