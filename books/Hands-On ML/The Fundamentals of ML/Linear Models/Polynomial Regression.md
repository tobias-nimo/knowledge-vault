# Polynomial Regression
Surprisingly, you can use a [[Linear Regression|linear model]] to fit nonlinear data: **add powers of each feature as new features, then train a linear model on this extended set of features**. This technique is called #polynomial-regression.

First, let's generate some nonlinear, noisy data based on a simple quadratic equation — one of the form $y = ax^2 + bx + c$:

```python
import numpy as np

rng = np.random.default_rng(seed=42)
m = 200 # number of instances
X = 6 * rng.random((m, 1)) - 3
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))
```

![[4-12.png]]

Use Scikit-Learn's `PolynomialFeatures` class to transform the training data, adding the square of each feature as a new feature — here there is just one feature:

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

X[0]       # array([1.64373629])
X_poly[0]  # array([1.64373629, 2.701869])

# `X_poly` now contains the original feature plus its square
```

You can then fit a plain `LinearRegression` model to this extended data:

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# (array([2.00540719]), array([[1.11022126, 0.50526985]]))
```

![[4-13.png]]

Not bad: the model estimates $\hat{y} = 0.56 x_1^2 + 0.93 x_1 + 1.78$, when the original function was $y = 0.5 x_1^2 + 1.0 x_1 + 2.0$ plus Gaussian noise.

>[!note] Finding relationships between features
>When there are multiple features, polynomial regression is capable of **finding relationships between features**, something a plain linear regression model cannot do. This works because `PolynomialFeatures` also **adds all combinations of features up to the given degree**.
>
> For example, with two features $a$ and $b$, `PolynomialFeatures(degree=3)` adds not only $a^2$, $a^3$, $b^2$, and $b^3$, but also the cross terms $ab$, $a^2 b$, and $a b^2$.

> [!warning] Combinatorial explosion
> `PolynomialFeatures(degree=d)` transforms an array of $n$ features into an array of $\dfrac{(n + d)!}{d!\,n!}$ features. **Beware of the combinatorial explosion of the number of features!**
