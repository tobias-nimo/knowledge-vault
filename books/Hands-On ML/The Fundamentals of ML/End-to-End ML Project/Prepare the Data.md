# Prepare the Data for Machine Learning Algorithms
Machine Learning algorithms often benefit from **transformed data that better highlights the underlying patterns**. Instead of applying these transformations manually, implement them as **reusable functions** or pipelines.

First, revert to a clean training set. You should also **separate the predictors and the labels**, since you don’t necessarily want to apply the same transformations to the predictors and the target values.
```python
features = train_set.drop("target", axis=1)
labels = train_set["target"].copy()
```
## Clean the Data
Most machine learning algorithms cannot work with **missing features**, so you’ll need to take care of these. You have three options to fix this:
1. Get rid of the instances.
2. Get rid of the whole attribute.
3. Set the missing values to some value (zero, the mean, the median, etc.). This is
called imputation.
```python
# Drop instances (option 1)
features.dropna(subset=["var1"], inplace=True) 

# Drop the whole attribute (option 2)
features.drop("var1", axis=1, inplace=True) 
```
To perform **imputation**, it is recommended to use Scikit-Learn `SimpleImputer` class. The imputer learns a replacement value for each feature in the training set (e.g., the median) and stores it internally. This allows the exact same values to be used later when imputing missing data in the validation set, test set, or any new data fed to the model.
```python
from sklearn.impute import SimpleImputer

# Create an imputer that replaces missing values with the median (option 3)
imputer = SimpleImputer(strategy="median") 

# The median can only be computed on numerical attributes
features_num = features.select_dtypes(include=[np.number])

# Learn the median of each numerical feature
imputer.fit(features_num) 

print(imputer.statistics_) # array([-118.51 , 34.26 , 29. , ...])
print(features_num.median().values) # array([-118.51 , 34.26 , 29. , ...])

# Replace missing values using the learned medians  
X = imputer.transform(features_num) # output is a NumPy array!

# Scikit-Learn transformers return NumPy arrays (even when they are fed dfs)
# Wrap the output in a DataFrame to restore the original structure
features_tr = pd.DataFrame(
	X,
	columns=features_num.columns,
	index=features_num.index
)
```
> There are many more Scikit-Learn [[Imputers]] available in the `sklearn.impute` package.
## Handling Text and Categorical Attributes
Suppose our dataset has one text-based feature, and it is categorical.
```python
features_cat = features[["cat_var"]]
features_cat.value_counts()
```
Most machine learning algorithms prefer to work with numbers, so let’s **convert these categories from text to numbers**. 
### Ordinal Encoding
#ordinal-encoding **converts each category into an integer value**. For this, we can use Scikit-Learn’s `OrdinalEncoder` class:
```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
features_cat_encoded = ordinal_encoder.fit_transform(features_cat)

print(features_cat_encoded) # array([[3.], [0.], [1.], ...])
print(ordinal_encoder.categories_) # [array(['text1', 'text2', ...], dtype=object)]
```
One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. This may be fine for **ordinal categories** (e.g., "bad", "average", "good", and "excellent"), but not for **nominal categories** (e.g., "red", "blue" and "green").
### One-Hot Encoding
#one-hot-encoding  solves this problem by creating **one binary feature for each category**. For example, a feature with categories `red`, `green`, and `blue` can be transformed into three binary features:

| color | red | green | blue |
| ----- | --- | ----- | ---- |
| red   | 1   | 0     | 0    |
| green | 0   | 1     | 0    |
| blue  | 0   | 0     | 1    |
The resulting binary features are often called **dummy variables**: takes the value `1` if the instance belongs to that category and `0` otherwise.

> If a feature has a very large number of categories, one-hot encoding **can create many new features, increasing memory usage and training time**. In such cases, alternative encodings or learned embeddings may be more appropriate.

**Scikit-Learn** provides the `OneHotEncoder` transformer to automatically convert categorical features into one-hot encoded vectors.
```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
features_cat_1hot = cat_encoder.fit_transform(features_cat)

features_cat_1hot # SciPy sparse matrix
features_cat_1hot.toarray() # (dense) NumPy array

print(cat_encoder.categories_) # [array(['red', 'green', 'blue'], dtype=object)]
```
`OneHotEncoder` remembers the categories seen during training and always produces the same output columns in the same order. This **ensures that new data is transformed consistently**, which is essential when deploying machine learning models. It can also detect previously unseen categories (or ignore them if `handle_unknown="ignore"` is set).  
```python
df_output = pd.DataFrame(
	cat_encoder.transform(df_new),
	columns=cat_encoder.get_feature_names_out(),
	index=df_new.index
)
```
## Feature Scaling and Transformation
One of the most important transformations you need to apply to your data is feature scaling. With few exceptions, **machine learning algorithms don’t perform well when the input numerical attributes have very different scales**.

There are two common ways to get all attributes to have the same scale: 
### Normalization
#normalization (or min-max scaling) **rescales values to a fixed range**, usually 0–1:
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
Scikit-Learn provides the `MinMaxScaler` transformer for this task. The `feature_range` hyperparameter can be used to specify a different output range.
```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
features_num_norm = min_max_scaler.fit_transform(features_num)
```
### Standardization
#standardization rescales values so they have a **mean of 0 and a standard deviation of 1**:  
$$x' = \frac{x - \mu}{\sigma}$$Unlike normalization, standardization does not restrict values to a fixed range, but it is generally **less sensitive to outliers**. Scikit-Learn provides the `StandardScaler` transformer for this task.
```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
features_num_std = std_scaler.fit_transform(features_num)
```

> Always remember to **fit scalers on the training set only** to avoid data leakage. After fitting, use `transform()` on the validation set, test set, and new data. With `MinMaxScaler`, values outside the training range may be scaled outside the target range; set `clip=True` to prevent this.

> Check this notes about **[[Handling Non-Symmetrical Distributions]] before scaling**.
### Transforming the Target Variable
Sometimes the target variable also requires transformation. For example, if the target has a heavy-tailed distribution, applying a logarithmic transformation may improve model performance.

**After making predictions, the inverse transformation must be applied to recover values on the original scale**. Scikit-Learn provides `TransformedTargetRegressor`, which automatically applies a transformation to the target during training and reverses it during prediction:
```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model = TransformedTargetRegressor(
	LinearRegression(),
	transformer=StandardScaler()
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
This is usually safer and less error-prone than manually transforming and inverse-transforming the target values.
## Custom Transformers
Although Scikit-Learn provides many useful transformers, you will occasionally need to **write your own** [[Custom Transformers]] for specific tasks.
## Transformation Pipelines
Real-world preprocessing often requires **multiple transformations applied in a specific order**. Scikit-Learn's `Pipeline` class allows you to chain these steps into a single estimator.

For example, a typical numerical pipeline might be:
```python  
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler  
  
num_pipeline = Pipeline([
("impute", SimpleImputer(strategy="median")), # step 1: impute missing values
("standardize", StandardScaler()), # step 2: scale the features
])

# pipelines support indexing; for example, pipeline[1] returns the second estimator
```
Or, you can use the `make_pipeline()` function instead:
```python
from sklearn.pipeline import make_pipeline

# make_pipeline automatically generate step names from the transformer class names (in lowercase and without underscores)
# if multiple transformers have the same name, an index is appended to their names (e.g., "foo-1", "foo-2", etc.).

num_pipeline = make_pipeline(
	SimpleImputer(strategy="median"), # simpleimputer
	StandardScaler() # standardscaler
	)
	
# num_pipeline["simpleimputer"] returns the estimator named "simpleimputer".
```

The estimators must all be transformers (i.e., they must have a `fit_transform()` method), except for the last one, which can be anything: a transformer, a predictor, or any other type of estimator.

When you call the pipeline’s `fit()` method, it calls `fit_transform()` sequentially on all the transformers, passing the output of each call as the parameter to the next call until it reaches the final estimator, for which it just calls the `fit()` method.

The pipeline exposes the same methods as the final estimator: 
- If the last estimator is a transformer, the pipeline also acts like a transformer. If you call the pipeline’s `transform()` method, it will sequentially apply all the transformations to the data.
- If the last estimator is a a predictor, then the pipeline would have a `predict()` method instead. Calling it would sequentially apply all the transformations to the data and pass the result to the predictor’s `predict()` method.

Let’s apply this pipeline to some training data:
```python
features_num_prepared = num_pipeline.fit_transform(features_num)

features_num_prepared.round(2)
# array([[-1.42, 1.01, 1.86, 0.31, 1.37, 0.14, 1.39, -0.94],
# [ 0.6 , -0.7 , 0.91, -0.31, -0.44, -0.69, -0.37, 1.17], ...])

# If you want to recover a nice DataFrame:
df_features_num_prepared = pd.DataFrame(
	features_num_prepared,
	columns=num_pipeline.get_feature_names_out(),
	index=features_num.index
	)
```











