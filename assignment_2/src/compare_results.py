# compare_results.py
# Aggregates and formats results across all architectures/datasets into a single readable report.
# Same filename/role as Assignment 1, extended to handle multiple architectures per dataset.

# generate_comparison_report()
#   - Parses the saved per-architecture metrics files (now including a hidden-layer-shape token,
#     e.g. H8 or H16-H8, alongside activation/learning-rate/epoch settings), and writes a
#     consolidated table with separate validation and test rows per architecture, highlighting the
#     best-selected configuration for each dataset.
