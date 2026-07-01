# Learning Curves
How can you tell whether your model is #overfitting or #underfitting the data?
- #cross-validation gives an estimate of generalization performance: a model that scores well on the training data but poorly in cross-validation is overfitting; one that scores poorly on both is underfitting.
- #learning-curves plot the model's **training error and validation error as a function of the training set size** (or training iteration).

Scikit-Learn's `learning_curve()` function trains and evaluates a model using cross-validation, retraining on growing subsets. The function returns the training set sizes at which it evaluated the model, plus the training and validation scores for each size and each cross-validation fold.

> [!tip] Incremental learning
> If the model supports incremental learning, set `exploit_incremental_learning=True` when calling `learning_curve()` and it will train the model incrementally instead of retraining from scratch on each subset.

First, let's generate some nonlinear, noisy data based on a simple quadratic equation — one of the form $y = ax^2 + bx + c$:

```python
import numpy as np

rng = np.random.default_rng(seed=42)
m = 200 # number of instances
X = 6 * rng.random((m, 1)) - 3
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))
```

![[4-12.png]]

Now, let's look at the learning curves of the plain [[Linear Regression|linear regression]] model:

```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.show()
```

![[4-15.png]]
> These curves are **typical of an underfitting model:** both reach a plateau, they're close together, and they're fairly high.

> [!warning] More data won't help an underfitting model
> If your model is underfitting, **adding more training examples will not help**. You need a better model or better features.

Now the learning curves of a 10th-degree polynomial model on the same data, built with a [[Polynomial Regression|polynomial]] pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")

# ...same plotting as before
```

![[4-16.png]]
> These curves are **typical of an overfitting model:** the training error is low and there is a gap between the curves — the model performs better on the training data than on the validation data.

> [!tip] Fixing overfitting
> One way to improve an overfitting model is to **feed it more training data** until the validation error reaches the training error.
