# Deep Learning Assignment 1: Complete Documentation

## 1. Purpose and Scope

This project implements a small neural-learning toolkit using only NumPy and
Matplotlib. It demonstrates:

- A single-neuron perceptron trained with full-batch gradient descent.
- Binary classification with logistic and hyperbolic-tangent activations.
- Multi-class classification through one-vs-one (OvO) decomposition.
- Univariate and bivariate linear regression with the linear activation.
- Data loading, deterministic train/test splitting, evaluation metrics, and
	experiment plotting.
- Automatic collection of the saved metrics into a comparison report.

The implementation is intentionally explicit. The model, activation functions,
confusion matrix, classification metrics, regression metrics, data splitting,
and plotting routines are implemented in the project rather than delegated to
scikit-learn.

## 2. Repository Layout

The executable code is in `src/`. The scripts expect to be run with `src` as
the current working directory because all data and output paths are relative
to that directory.

```text
src/
|-- classification.py       Classification experiment driver
|-- compare_results.py      Saved-metric report generator
|-- multiclass.py           One-vs-one multi-class wrapper
|-- perceptron.py           Single-layer model and training loop
|-- regression.py           Regression experiment driver
|-- requirements.txt        Python dependencies
|-- data/
|   |-- Classification/
|   |   |-- LS_Group30/      Three class files for linearly separable data
|   |   `-- NLS_Group30.txt  Non-linear dataset
|   `-- Regression/
|       |-- UnivariateData/  One input feature plus target
|       `-- BivariateData/   Two input features plus target
|-- results/                 Generated plots and metric files
|-- utils/
		|-- activations.py       Activation functions and derivatives
		|-- data_utils.py        Loaders and split functions
		|-- metrics.py           Classification and regression metrics
		`-- plotting.py          All Matplotlib visualizations
```

`utils/__init__.py` makes `utils` importable as a package. The `utils/results`
directory is present in the repository but is not used by the experiment
scripts; generated outputs are written to the top-level `src/results` folder.

## 3. Dependencies and Execution Assumptions

The required packages are:

- Python 3.10 or later is recommended by the setup guide.
- `numpy>=1.24` for arrays, vectorized arithmetic, permutations, and numerical
	calculations.
- `matplotlib>=3.7` for all plots, including the 3D regression visualization.

Install them from inside `src`:

```powershell
python -m pip install -r requirements.txt
```

Run the workflow in this order:

```powershell
python classification.py
python regression.py
python compare_results.py
```

The scripts do not accept command-line arguments. Their experiment settings
are constants at the top of the corresponding script.

## 4. Mathematical Model

### 4.1 Single-layer neuron

For an input matrix $X$ with $n$ samples and $d$ features, the model stores a
weight vector $W \in \mathbb{R}^d$ and scalar bias $b$. The forward pass is:

$$
z_i = X_i W + b
$$

$$
\hat{y}_i = f(z_i)
$$

The implementation computes this for every sample at once with
`np.dot(X, self.weights) + self.bias`.

### 4.2 Full-batch gradient descent

The training error used by the model is the mean squared error of the model
output:

$$
E = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

The residual is $e_i=\hat{y}_i-y_i$. Applying the chain rule gives:

$$
\delta_i = e_i f'(z_i)
$$

$$
\nabla_W E = \frac{1}{n}X^T\delta,
\qquad
\nabla_b E = \frac{1}{n}\sum_{i=1}^{n}\delta_i
$$

The update is performed once per epoch using the entire training set:

$$
W \leftarrow W-\eta\nabla_W E,
\qquad
b \leftarrow b-\eta\nabla_b E
$$

where $\eta$ is `learning_rate`. There is no mini-batching, shuffling inside
the training loop, momentum, regularization, early stopping, or parameter
normalization. Weights start at zero and the bias starts at zero every time
`fit` is called.

## 5. `perceptron.py`

### `Perceptron.__init__(learning_rate=0.01, epochs=1000, activation="logistic")`

Creates the model and validates the activation name against
`utils.activations.ACTIVATIONS`. An invalid name raises `ValueError` and lists
the accepted names. It stores:

- `learning_rate`: scalar step size used by gradient descent.
- `epochs`: number of complete passes through the supplied training data.
- `activation_name`: the requested string, used later by OvO thresholding.
- `activation`: callable selected from the activation registry.
- `activation_derivative`: corresponding derivative callable.
- `weights` and `bias`: initially `None` until training.
- `errors`: one MSE value per completed epoch.

### `_initialize_parameters(n_features)`

Allocates a floating-point zero vector of length `n_features` and sets the
bias to `0.0`. This is called at the beginning of every `fit`, so retraining a
model replaces rather than continues its previous parameters.

### `_forward(X)`

Computes and returns a tuple `(z, y_pred)`:

1. `z = X dot weights + bias` is the pre-activation value.
2. `y_pred = activation(z)` is the output after applying the selected
	 activation elementwise.

The method is internal but is also reused by `predict`.

### `fit(X, y)`

Converts both inputs to floating-point NumPy arrays. Targets are flattened to
one dimension with `reshape(-1)`. It validates that `X` is two-dimensional and
that the number of samples in `X` and `y` matches.

The method initializes parameters, clears the old error history, then repeats
the following for `epochs` iterations:

1. Run the forward pass.
2. Compute `error = y_pred - y`.
3. Compute the activation derivative at `z`.
4. Multiply residual and derivative to obtain `delta`.
5. Compute the averaged weight and bias gradients.
6. Apply the gradient-descent update.
7. Compute and append the post-forward residual MSE to `errors`.

It returns `self`, enabling calls such as
`model = Perceptron(...).fit(X_train, y_train)`.

### `predict(X)`

Converts `X` to a floating-point array and raises `ValueError` if `fit` has
not initialized the weights. It returns continuous activation outputs from a
forward pass. It does not convert outputs to class labels; classification
thresholding is performed by `OneVsOneClassifier`.

## 6. `utils/activations.py`

The activation registry maps each name to a pair `(activation, derivative)`.
Both the canonical name `linear` and the alias `identity` select the same
functions.

### Linear / identity

$$
f(z)=z, \qquad f'(z)=1
$$

`linear` converts its input to floating point and returns it unchanged.
`linear_derivative` returns an array of ones with the same shape. This gives a
linear model suitable for regression:

$$
\hat{y}=XW+b
$$

### Logistic / sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}},
\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

The output range is $(0,1)$. The implementation handles positive and negative
values separately: non-negative values use $1/(1+e^{-z})$, while negative
values use $e^z/(1+e^z)$. This avoids unnecessarily evaluating a large
$e^{-z}$ for negative inputs and reduces exponential overflow risk.

`logistic_derivative_from_output` provides the equivalent derivative when the
already-computed output is available, but the current `Perceptron` training
loop calls `logistic_derivative(z)` instead.

### Hyperbolic tangent

$$
f(z)=\tanh(z),
\qquad
f'(z)=1-\tanh(z)^2
$$

The output range is $(-1,1)$. `tanh_derivative_from_output` computes the same
derivative from an existing output. Like the logistic output helper, it is
available for reuse but is not used by the current training loop.

## 7. `multiclass.py`: One-vs-One Classification

`OneVsOneClassifier` turns binary `Perceptron` instances into a multi-class
classifier.

### Initialization

The constructor stores the activation, learning rate, and epoch count and
initializes an empty `classifiers` dictionary and `classes_ = None`.

### `_decision_threshold(model)`

The binary output is converted to a hard decision using:

- `0.5` for a logistic model, because sigmoid output represents a probability-
	like value between zero and one.
- `0.0` for a tanh model, because tanh is centered around zero.

Outputs below the threshold map to the first class in the pair; outputs at or
above it map to the second class.

### `fit(X, y)`

The method converts `X` to floating point, records sorted unique labels using
`np.unique`, and creates one binary model for every unordered pair generated by
`itertools.combinations`.

For each pair `(class_a, class_b)`:

1. Select only samples whose label is `class_a` or `class_b`.
2. Recode `class_a` as binary target `0.0` and `class_b` as `1.0`.
3. Construct a fresh `Perceptron` with the requested settings.
4. Train it on that pair's samples.
5. Store it under `classifiers[(class_a, class_b)]`.

For $K$ classes, this creates $K(K-1)/2$ binary models. With three classes,
the pairs are `(0, 1)`, `(0, 2)`, and `(1, 2)`.

### `predict_pair_labels(class_a, class_b, X)`

Runs the stored pairwise model, thresholds its raw output, and returns original
class labels rather than binary `0`/`1` values. This function is used by the
pair-specific decision-region plots.

### `predict(X)`

Raises `ValueError` before fitting. Otherwise it creates an integer vote matrix
with one column per class and one row per input sample. Every pairwise model
casts one vote for one of its two classes. `np.argmax` selects the class with
the largest vote count for each sample. Ties are resolved by NumPy's first
maximum, which corresponds to the first class in `classes_`.

## 8. `utils/data_utils.py`

### `train_test_split(X, y, test_ratio=0.3, seed=42, rng=None)`

Creates a random permutation of sample indices. The first
`int(len(X) * test_ratio)` indices become the test set and the remaining
indices become the training set. It returns:

```text
X_train, X_test, y_train, y_test
```

When `rng` is omitted, a local `np.random.default_rng(seed)` is created. When
provided, the supplied generator is used and its state advances. The global
NumPy random state is never changed.

### `stratified_train_test_split(X, y, test_ratio=0.3, seed=42)`

Creates one seeded generator and splits each unique class independently with
`train_test_split(..., rng=rng)`. The class-specific parts are then combined
with `np.vstack` and `np.concatenate`. This preserves approximately the same
test ratio in every class, unlike a single unstratified permutation.

The returned class groups are concatenated by class and are not reshuffled
after concatenation. Classification uses this function; regression uses the
ordinary unstratified split.

### `load_LS_data(par_dir)`

Reads exactly `Class1.txt`, `Class2.txt`, and `Class3.txt`. Each non-empty line
with exactly two whitespace-separated values is interpreted as two floating-
point features. Labels are assigned by file order: `Class1.txt` becomes `0`,
`Class2.txt` becomes `1`, and `Class3.txt` becomes `2`. It returns NumPy arrays
`(X, y)`.

### `load_nls_data(filepath)`

Reads all lines, skips the first line unconditionally (the expected header),
and accepts non-empty lines containing at least two fields. The first two fields
are the features. Labels are assigned by row position after the header:

- rows with index `0` through `499` receive label `0`;
- rows with index `500` through `999` receive label `1`;
- all later accepted rows receive label `2`.

Therefore this loader assumes the file is ordered in three blocks of samples;
it does not read a label column from the file.

### `load_regression_csv(filepath)`

Reads comma-separated rows. It tests the first field of the first line: if it
cannot be converted to `float`, that line is treated as a header and skipped.
For every remaining non-empty row, all fields except the last become `X` and
the last field becomes `y`. This supports both one-feature and two-feature
regression files without separate parsing logic.

## 9. `utils/metrics.py`

### `confusion_matrix(y_true, y_pred, num_classes)`

Creates a `num_classes` by `num_classes` integer matrix. Rows are true labels,
columns are predicted labels. For each pair `(t, p)`, the implementation
increments `matrix[int(t)][int(p)]`.

### `classification_metrics(...)`

First builds the confusion matrix, then computes one-vs-rest quantities for
each class $i$:

$$
TP_i=CM_{i,i},\quad
FP_i=\sum_r CM_{r,i}-TP_i,\quad
FN_i=\sum_c CM_{i,c}-TP_i
$$

$$
TN_i=N-(TP_i+FP_i+FN_i)
$$

It returns:

- `overall_accuracy`: trace of the confusion matrix divided by total samples.
- `class_accuracy`: $(TP_i+TN_i)/N$ for each class.
- `class_precision`: $TP_i/(TP_i+FP_i)$, or `0.0` if undefined.
- `class_recall`: $TP_i/(TP_i+FN_i)$, or `0.0` if undefined.
- `class_f_measure`: harmonic mean of precision and recall, or `0.0` when both
	are zero.
- `macro_accuracy`, `macro_precision`, `macro_recall`, and `macro_f_measure`:
	arithmetic means across classes.
- `micro_precision`, `micro_recall`, and `micro_f_measure`: metrics calculated
	from summed one-vs-rest counts.
- `confusion_matrix`: the raw integer matrix.

`print_classification_report` formats these values for the console, but the
classification experiment currently writes the dictionary directly to a file
and does not call this printer.

### Regression metrics

`mse` computes:

$$
MSE=\frac{1}{N}\sum_i(y_i-\hat{y}_i)^2
$$

`rmse` returns $\sqrt{MSE}$. `percent_rmse` divides RMSE by the absolute mean
of the true targets and multiplies by 100:

$$
\%RMSE=\frac{RMSE}{|\operatorname{mean}(y)|}\times100
$$

This percentage is relative to the target mean, not to a standard deviation,
range, or per-sample percentage error.

## 10. `utils/plotting.py`

The module imports Matplotlib lazily inside a guarded import block and raises a
clear installation error if Matplotlib is unavailable. It uses the
`seaborn-v0_8-whitegrid` style and the `tab10` palette. Every plotting function
saves a PNG at 150 DPI with a tight bounding box and closes its figure.

### `plot_error_vs_epochs`

Plots the supplied error history against epoch numbers `1` through
`len(errors)`. It uses an 8 by 5 inch figure, circular markers, a blue line,
axis labels, a title, and `tight_layout`. For the perceptron, these values are
the post-update MSE recorded after each full-batch epoch.

### `plot_decision_regions`

This is a two-dimensional decision visualization:

1. It expands the feature bounds by one unit on each side.
2. It creates a mesh grid with spacing `0.05`.
3. It calls the supplied `predict_fn` on every grid point.
4. It reshapes predictions to the grid shape and fills the regions with
	 `contourf`.
5. It overlays training points with black edges and adds a class legend.

`all_classes` controls stable colors. The helper `_class_color_map` sorts all
class labels and assigns them colors from `tab10`, so a class keeps the same
color in pairwise and combined plots. If `all_classes` is omitted, only the
classes present in the plotted data are used.

This function requires at least two feature columns because it indexes columns
zero and one. Classification in this project always supplies two-dimensional
data.

### `plot_regression_1d`

Plots true target points as blue scatter points and model outputs as a red
line. The input values are sorted before drawing the model line so the line
does not zigzag according to the original sample order. It is used for both
training and test subsets of the univariate dataset.

### `plot_regression_2d`

Creates a 3D Matplotlib axis. True targets and model outputs are shown as two
colored point clouds at the same `(x1, x2)` coordinates. The axes are labeled
`x1 values`, `x2 values`, and `y-value`. It is used for bivariate train and test
subsets.

### `plot_target_vs_model_scatter`

Plots true targets on the horizontal axis and predictions on the vertical
axis. It computes the common minimum and maximum across both arrays and draws
the dashed ideal-fit line $y=x$. Equal axis scaling makes deviations from the
ideal line visually comparable. Separate train and test plots are produced.

## 11. Classification Experiment Driver

`classification.py` defines:

```text
ACTIVATIONS   = ["logistic", "tanh"]
LEARNING_RATES = [0.01, 0.1]
EPOCHS_LIST   = [500, 1000]
TEST_RATIO    = 0.3
SEED          = 42
```

### `run_dataset(dataset_name, X, y, activation, lr, epochs)`

The function creates an output directory named
`results/<dataset>/<activation>_LR<lr>_EP<epochs>`, performs a stratified 70/30
split, and trains one `OneVsOneClassifier`.

It then creates:

- one error-history plot for every pairwise classifier;
- one pairwise decision-region plot for every class pair;
- one combined decision-region plot for the full classifier;
- `evaluation_metrics.txt` containing test-set metrics.

The metrics are calculated from `clf.predict(X_test)` and
`classification_metrics(y_test, y_pred_test, num_classes)`. The file includes
the configuration line and the string representation of every returned metric,
including the confusion matrix and all class, macro, and micro values.

### `main()`

Loads LS and NLS data, then runs the Cartesian product of both datasets, two
activations, two learning rates, and two epoch counts: $2\times2\times2\times
2=16$ classification configurations. It returns a dictionary keyed by
`(dataset_name, activation, lr, epochs)`, although the command-line execution
does not use that returned value.

## 12. Regression Experiment Driver

`regression.py` defines two datasets:

- `Univariate`: one feature, loaded from `data/Regression/UnivariateData/30.csv`.
- `Bivariate`: two features, loaded from `data/Regression/BivariateData/30.csv`.

The regression settings are:

```text
LEARNING_RATES = [0.01, 0.05]
EPOCHS_LIST   = [500, 1000]
TEST_RATIO    = 0.3
SEED          = 42
```

### `report_rmse(y_true, y_pred)`

Small convenience function returning `(rmse, percent_rmse)`.

### `run_dataset(dataset_name, path, dim, lr, epochs)`

Loads the CSV, performs a deterministic unstratified 70/30 split, and trains a
`Perceptron` using the `linear` activation. It saves:

- `error_vs_epoch.png`;
- `evaluation_metrics.txt` with train and test RMSE and %RMSE;
- `target_vs_model_train.png`;
- `target_vs_model_test.png`;
- `scatter_train.png`;
- `scatter_test.png`.

For `dim == 1`, the target/model plot uses `plot_regression_1d`; otherwise it
uses `plot_regression_2d`. Predictions are generated for both subsets before
metrics and plots are produced. The function returns the trained model.

### `main()`

Runs the two datasets across two learning rates and two epoch counts, producing
$2\times2\times2=8$ regression configurations. It returns a dictionary keyed
by `(dataset_name, lr, epochs)`.

## 13. Output Organization

After all experiments, classification produces 16 directories and regression
produces 8 directories:

```text
results/
|-- LS/
|   `-- logistic_LR0.01_EP500/
|       |-- decision_class0_vs_class1.png
|       |-- decision_class0_vs_class2.png
|       |-- decision_class1_vs_class2.png
|       |-- decision_combined.png
|       |-- error_class0_vs_class1.png
|       |-- error_class0_vs_class2.png
|       |-- error_class1_vs_class2.png
|       `-- evaluation_metrics.txt
|-- NLS/                     Same classification artifact pattern
|-- Univariate/
|   `-- linear_LR0.01_EP500/
|       |-- error_vs_epoch.png
|       |-- evaluation_metrics.txt
|       |-- scatter_test.png
|       |-- scatter_train.png
|       |-- target_vs_model_test.png
|       `-- target_vs_model_train.png
`-- Bivariate/               Same regression artifact pattern
```

The exact learning-rate and epoch directory names vary with the configuration.
Existing files with the same names are overwritten when an experiment is run.

## 14. Comparison Report Generation

`compare_results.py` recursively searches for every
`results/**/evaluation_metrics.txt`. It parses each file's configuration line
and then extracts either:

- `overall_accuracy` for classification; or
- `Test RMSE` and `Test %RMSE` for regression.

Classification is identified when the parsed activation is `logistic` or
`tanh`. Entries are grouped into LS/NLS and activation sections. Regression
entries are grouped into Univariate and Bivariate sections. Each group is
sorted lexicographically as a formatted string before being written to
`results_comparison_report.txt`.

The report is a summary only. It does not recompute metrics, inspect plots, or
select a statistically optimal configuration. If no evaluation files are
found, the script prints a message and returns without creating a new report.

## 15. Current Results and Interpretation

The checked-in comparison report records these broad observations:

- Logistic activation performs strongly on the LS dataset, where a linear
	decision boundary is appropriate.
- Both logistic and tanh versions are weaker on NLS because each pairwise
	classifier is still a single linear neuron; changing the activation does not
	add hidden layers or nonlinear input features.
- The logistic LS configurations have the highest recorded classification
	accuracies in the supplied report.
- Increasing epochs generally improves the recorded regression results, while
	the learning-rate effect depends on the dataset and configuration.
- Bivariate regression has larger absolute RMSE values than univariate
	regression in the supplied report; RMSE remains in the scale of the target.

These are observations of the checked-in run, not guarantees for changed data,
random seeds, or numerical environments.

## 16. Important Assumptions and Limitations

- Relative paths are resolved from the current working directory. Run scripts
	from `src`.
- `load_nls_data` assumes a one-line header and fixed positional class blocks.
- `load_LS_data` ignores malformed lines instead of reporting them.
- Classification labels must be usable as integer indices from `0` through
	`num_classes - 1` for `confusion_matrix` and vote indexing.
- The classifier assumes every class pair has at least one sample.
- `plot_decision_regions` is specific to two input dimensions.
- The model uses unscaled features. Large feature magnitudes can make gradient
	descent unstable or cause activation saturation.
- Training minimizes output MSE for every activation. It does not use binary
	cross-entropy for logistic classification.
- For tanh classification, targets are still recoded as `0.0` and `1.0`, while
	the decision threshold is `0.0`. This follows the current implementation but
	is not the usual centered tanh target convention of `-1` and `+1`.
- `percent_rmse` is undefined when the mean target is zero; the current code
	does not add a special zero-denominator guard.
- `overall_accuracy` divides by the confusion-matrix total and therefore
	assumes a non-empty evaluation set.
- There is no persistence format for trained weights. Running an experiment
	reconstructs and retrains all models.
- The imported `print_classification_report` function is available but is not
	invoked by `classification.py`.

## 17. End-to-End Data Flow

```text
Input files
		|
		v
Data loaders -> NumPy feature matrix X and target/label vector y
		|
		v
Deterministic train/test split (stratified for classification)
		|
		+--> Perceptron(s) -> forward pass -> activation -> full-batch gradients
		|                         |
		|                         +--> errors history -> error plot
		|
		+--> OvO vote aggregation for classification
		|
		v
Predictions on train/test data
		|
		+--> classification metrics or RMSE metrics -> evaluation_metrics.txt
		+--> decision/regression plots -> PNG artifacts
		|
		v
compare_results.py -> results_comparison_report.txt
```

The central design is therefore a reusable binary/vector regression model,
with dataset-specific drivers responsible for loading data, selecting
hyperparameters, evaluating predictions, and saving artifacts.
