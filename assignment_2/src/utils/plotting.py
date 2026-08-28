# utils/plotting.py
# Additions on top of Assignment 1's version — adds plotting functions for visualizing individual
# hidden/output node responses, required by Assignment 2's presentation-of-results section.
# plot_error_vs_epochs, plot_decision_regions, plot_regression_1d/2d, and
# plot_target_vs_model_scatter are carried over unchanged from Assignment 1.

# plot_node_output_surface(X, z, node_label, split_name, filename)
#   - 3D scatter plot (x1, x2, node output) for a single hidden or output node, for 2D-input
#     datasets (LS/NLS classification, bivariate regression), across train/val/test splits.
#
# plot_node_output_1d(X, z, node_label, split_name, filename)
#   - 2D line/scatter plot (x, node output) for a single node, for the univariate regression
#     dataset.
