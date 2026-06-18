# Explore Different Models
The goal at this stage is to try a variety of models from different categories of ML algorithms (e.g., linear and polynomial regression, support vector machines with different kernels, ensemble methods, and possibly neural networks), **without spending too much time tuning hyperparameters**. The objective is to **shortlist of the most promising candidates**—typically two to four models—for further optimization.

> [!note] Remember from previous step:
> - You chose #RMSE as your performance metric — [[Look at the Big Picture]]
> - You built a **preprocessing pipeline** — [[Prepare the Data]]
## Train and Evaluate on the Training Set
Suppose you decide to start with a simple **linear regression** model:

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from my_pipelines import preprocessing # defined in ./prepare-the-data
from my_datafranes import features, labels # defined in ./prepare-the-data

# train the model
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(features, labels)

# take a pick at the predictions
predictions = lin_reg.predict(features)
predictions[:5] # array([246000., 372700., 135700., 91400., 330900.])
labels.iloc[:5].values # array([458300., 483800., 101700., 96100., 361800.])

from sklearn.metrics import root_mean_squared_error

# evaluate on the training set
lin_rmse = root_mean_squared_error(labels, predictions)
lin_rmse # 68972.88
```

This is not a great score. The model appears to be #underfitting the data. When this happens, it may indicate that the features do not provide enough predictive information or that **the model is not powerful enough** to capture the underlying patterns in the data.

Next, suppose you try a **decision tree** model:

```python
from sklearn.tree import DecisionTreeRegressor

# train the model
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(features, labels)

# evaluate on the training set
predictions = tree_reg.predict(features)
tree_rmse = root_mean_squared_error(labels, predictions) # 0.0
```

This result is not encouraging either. An RMSE of 0 on the training set strongly suggests that the model has **overfit** the data. #overfitting 

How can you verify that a model is overfitting the training data? Well, the key idea is to **evaluate it on data that was not used during training**...
## Evaluate on the Validation Set
One approach is to split the training set into a smaller training set and a validation set. You would then train the model on the smaller training set and evaluate it on the validation set.

```python
from sklearn.model_selection import train_test_split

# split the training data into a smaller training set and a validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

# train the model on the smaller training set
tree_reg = make_pipeline(
    preprocessing,
    DecisionTreeRegressor(random_state=42)
)
tree_reg.fit(X_train, y_train)

# evaluate on the validation set
predictions = tree_reg.predict(X_valid)
tree_rmse = root_mean_squared_error(y_valid, predictions) # 63837.60

# 0.0 <<< 63837.60 -> overfitting
```

This approach gives a **more realistic estimate of how the model will perform on unseen data**. However, the evaluation depends on a single train/validation split, which means the result may vary depending on which samples end up in the validation set.
## Cross-Validation
A more reliable alternative is **k-fold cross-validation**.

The training set is split into _k_ non-overlapping subsets called _folds_. The model is trained _k_ times. Each time, a different fold is used for validation while the remaining _k − 1_ folds are used for training. The final result is a collection of _k_ evaluation scores instead of a single score.
![[2-20.png]]

Scikit-Learn provides the `cross_val_score()` function to perform #cross-validation:

```python
from sklearn.model_selection import cross_val_score 

tree_rmses = -cross_val_score(
	tree_reg, features, labels,
	scoring="neg_root_mean_squared_error", cv=10
	)
	
# cross_val_score expects a score where higher is better (utility function), so RMSE (cost function) is returned as a negative value.
```

Let's inspect the results:

```python
import pandas as pd

pd.Series(tree_rmses).describe()
```

```
count       10.000000
mean     66573.734600
std       1103.402323
min      64607.896046
25%      66204.731788
50%      66388.272499
75%      66826.257468
max      68532.210664

In this case, the training error is extremely low (0.0) while the validation error is much higher, which is a classic sign of overfitting.
```

Cross-validation not only provides a real and **robust performance measure**, but also a **measure of uncertainty** in that estimate (the standard deviation).

> [!warning] The downside of cross-validation is that the model must be trained multiple times, which can become **computationally expensive** for large datasets or complex models.
## Try a More Powerful Model
Next, let's train a **random forest** model:

```python
from sklearn.ensemble import RandomForestRegressor

# try another model
forest_reg = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=42)
)

# evaluate on the training set
forest_reg.fit(features, labels)
predictions = forest_reg.predict(features)
root_mean_squared_error(labels, predictions) # ~17551

# evaluate on the val set using cv
forest_rmses = -cross_val_score(
    forest_reg,
    features,
    labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

pd.Series(forest_rmses).describe()
```

```
count       10.000000
mean     47038.092799
std       1021.491757
min      45495.976649
25%      46510.418013
50%      47118.719249
75%      47480.519175
max      49140.832210

This large gap between the training RMSE (~17551) and the cross-validation RMSE (~47038) indicates that the model is still overfitting the training data.
```

Although the model is not perfect, it performs significantly better than both linear regression and a single decision tree, making it our most promising candidate so far.