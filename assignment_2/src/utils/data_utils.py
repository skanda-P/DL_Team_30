# utils/data_utils.py
# Additions on top of Assignment 1's version — adds the 60/20/20 splitting utilities needed for
# Assignment 2's train/validation/test scheme. load_LS_data, load_nls_data, and load_regression_csv
# are carried over unchanged from Assignment 1.

# three_way_split(X, y, train_ratio, val_ratio, seed, rng)
#   - Splits a dataset into train/validation/test sets by chaining two applications of the existing
#     two-way split logic.
#
# stratified_three_way_split(X, y, train_ratio, val_ratio, seed)
#   - Same as above but preserves per-class proportions in each split, extending Assignment 1's
#     stratified two-way split.
#
# Note: load_nls_data should be checked against the real data file's class-count header before
# reuse — it currently assumes hardcoded class boundaries rather than reading the header.
