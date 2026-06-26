# Scikit-Learn API Design
[Scikit-Learn’s API](https://scikit-learn.org/stable/) is remarkably well designed. Its main design principles are:
## Consistency
All objects share a simple, consistent interface:
- **Estimators**
	Any object that learns parameters from data is an estimator (e.g., `SimpleImputer` and `LinearRegression`). Learning is performed with `fit()`, which takes a dataset (and labels for supervised learning). Configuration options that guide learning are called hyperparameters and are typically set through the constructor (e.g., `strategy`).
- **Transformers**
	Some estimators can transform data; these are called transformers (e.g., [[Imputers]]). They implement `transform()`, which returns the transformed dataset using the parameters learned during `fit()`. They also provide `fit_transform()`, which is equivalent to `fit()` followed by `transform()` and may be optimized for speed.
- **Predictors**
	Some estimators can make predictions; these are called predictors (e.g., `LinearRegression`). They implement `predict()`, which returns predictions for new instances, and `score()`, which evaluates prediction quality on a test set.
- **Inspection**
	Hyperparameters are accessible through public attributes (e.g., `imputer.strategy`), while learned parameters are stored in public attributes ending with an underscore (e.g., `imputer.statistics_`).

> [!note] Remember
> Scikit-Learn **estimators output NumPy arrays** (or sometimes SciPy sparse matrices) even when they are fed Pandas DataFrames as input.
## Non-proliferation of Classes
Datasets are represented using NumPy arrays or SciPy sparse matrices rather than custom dataset classes. Hyperparameters are ordinary Python values.
## Composition
Scikit-Learn reuses existing building blocks whenever possible. For example, multiple transformers and a final estimator can be combined into a `Pipeline`.
## Sensible Defaults
Most parameters have reasonable default values, making it easy to build a working baseline system quickly.