# compare_with_a1.py
# Compares Assignment 2's best FCNN results against Assignment 1's single-neuron perceptron results,
# as required by the assignment's "comparison of performance with the single neuron model" item.
# New file — Assignment 1 had nothing to compare against.

# load_a1_metrics(results_dir)
#   - Reads the saved evaluation metrics from Assignment 1's perceptron / one-vs-one runs.
#
# load_a2_best_metrics(results_dir)
#   - Reads the best-architecture FCNN metrics produced by classification.py / regression.py.
#
# write_comparison_table(a1_metrics, a2_metrics, out_path)
#   - Writes a side-by-side comparison table (accuracy for classification, RMSE/%RMSE for
#     regression) between the single-neuron and FCNN models, for inclusion in the report.
