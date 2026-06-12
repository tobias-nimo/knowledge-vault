# Scikit-Learn Custom Transformers
## Non-Trainable Transformers
For custom transformations that **don’t require any training**, you just use `FunctionTransformer` with a function that takes a NumPy array as input and outputs the transformed array.
```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# let’s create a log-transformer
log_transformer = FunctionTransformer(
	np.log,
	inverse_func=np.exp # optional, useful if you plan to transform a target
)

log_var = log_transformer.transform(features[["var"]])
```

```python
from sklearn.metrics.pairwise import rbf_kernel

# let’s measure the geographic similarity between each instance and some location
coords = 37.7749, -122.41
geo_transformer = FunctionTransformer(
	rbf_kernel,
	kw_args=dict(Y=[coords], gamma=0.1) # set hyperparameters as additional arguments
)

geo_simil = geo_transformer.transform(features[["latitude", "longitude"]])
```

```python
# if you want to combine features
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])

ratio_transformer.transform(np.array([[1., 2.], [3., 4.]])) # array([[0.5 ], [0.75]])
```
## Trainable Transformers
For custom trainable transformers that learn some parameters in the `fit()` method and use them later in the `transform()` method, you need to **write a custom class** implementing:
- `fit()`
- `transform()`
- `fit_transform()`

In practice, custom transformers usually inherit from:
- `TransformerMixin` → provides `fit_transform()`.
- `BaseEstimator` → provides `get_params()` and `set_params()`, making the transformer compatible with hyperparameter tuning.

For example, here’s a custom transformer that acts much like the `StandardScaler`:
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
	# avoid *args or **kwargs in the constructor, when using BaseEstimator
	def __init__(self, with_mean=True):
	self.with_mean = with_mean
	
	def fit(self, X, y=None): # y is required even though we don't use it!
	
		# every estimator should store feature_names_in_ in fit()
		if hasattr(X, "columns"):
			self.feature_names_in_ = np.asarray(X.columns)
		X = check_array(X) # checks for arrays with finite float values
		# learned attributes should end with trailing _
		self.mean_ = X.mean(axis=0)
		self.scale_ = X.std(axis=0)
		# every estimator should store feature_names_in_ in fit()
		self.n_features_in_ = X.shape[1]
		return self # always return self!
	
	def transform(self, X):
		check_is_fitted(self) # looks for learned attributes (with trailing _)
		X = check_array(X)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
			f"Expected {self.n_features_in_} features, got {X.shape[1]}"
			)
		if self.with_mean:
			X = X - self.mean_
		return X / self.scale_

	# implement `inverse_transform()` (when possible)	
	def inverse_transform(self, X):  
		check_is_fitted(self)
		X = check_array(X)
		X = X * self.scale_
		if self.with_mean:
			X = X + self.mean_
		return X

	# don't forget to implement `get_feature_names_out()`
	def get_feature_names_out(self, names=None):
		check_is_fitted(self)
		if hasattr(self, "feature_names_in_"):
			return [f"{name}_std" for name in self.feature_names_in_]
		return [f"x{i}_std" for i in range(self.n_features_in_)]
```

> Pay attention to the comments below—they illustrate **key Scikit-Learn transformer conventions**. 
> 
> You can **check whether your custom estimator respects Scikit-Learn’s API** by passing an instance to `check_estimator()` from the `sklearn.utils.estimator_checks` package.
## A Practical Example
A custom transformer can (and often does) **use other estimators in its implementation**.

For example, the transformer below:
1. Uses `KMeans` to **find geographic clusters** during `fit()`.
2. Uses Gaussian RBF to **measure similarity between each sample and each cluster center** during `transform()`.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
	def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
		self.n_clusters = n_clusters
		self.gamma = gamma
		self.random_state = random_state
	
	def fit(self, X, y=None, sample_weight=None):
		self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
		self.kmeans_.fit(X, sample_weight=sample_weight)
		return self # always return self!
	def transform(self, X):
		return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

	def get_feature_names_out(self, names=None):
		return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
		
# now let’s use it
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(features[["latitude", "longitude"]])

# the result is a matrix with one row per sample, and one column per cluster.
similarities.round(2)
# array([[0.46, 0. , 0.08, 0. , 0. , 0. , 0. , 0.98, 0. , 0. ],
# [0. , 0.96, 0. , 0.03, 0.04, 0. , 0. , 0. , 0.11, 0.35],
# [0.34, 0. , 0.45, 0. , 0. , 0. , 0.01, 0.73, 0. , 0. ], ...])
```
The following figure shows the 10 cluster centers found by k-means:
![[2-19.png]]








