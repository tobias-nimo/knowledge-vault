# Linear Regression
A #linear-regression model makes a prediction by **computing a weighted sum of the input features, plus a constant** called the *bias term* (or *intercept term*).
$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$
- $\hat{y}$ is the predicted value.
- $n$ is the number of features.
- $x_i$ is the $i^{\text{th}}$ feature value.
- $\theta_j$ is the $j^{\text{th}}$ model parameter — the bias term $\theta_0$ and the feature weights $\theta_1, \theta_2, \cdots, \theta_n$.

This can be written much more concisely in **vectorized form**:
$$
\hat{y} = h_\theta(\mathbf{x}) = \boldsymbol{\theta}^\intercal \cdot \mathbf{x}
$$
- $h_\theta$ is the **hypothesis function**, using the model parameters $\boldsymbol{\theta}$.
- $\boldsymbol{\theta}$ is the model's **parameter vector**, containing the bias term $\theta_0$ and the feature weights $\theta_1$ to $\theta_n$.
- $\mathbf{x}$ is the instance's **feature vector**, containing $x_0$ to $x_n$, with $x_0$ always equal to 1.
- $\boldsymbol{\theta} \cdot \mathbf{x}$ is the dot product $\theta_0 x_0 + \theta_1 x_1 + \cdots + \theta_n x_n$.

## Training: Minimizing the MSE
Training a model means **setting its parameters so that the model best fits the training set**. To do that, you first need a measure of how well (or poorly) it fits the data.

The most common performance measure for a regression model is the #RMSE (root mean squared error). So training a linear regression model means **finding the $\boldsymbol{\theta}$ that minimizes the RMSE**. In practice, it's **simpler to minimize** the #MSE (mean squared error), and it leads to the same result — the value that minimizes a positive function also minimizes its square root.

The MSE of a linear regression hypothesis $h_\theta$ on a training set $\mathbf{X}$ is:
$$
\text{MSE}(\mathbf{X}, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^\intercal \mathbf{x}^{(i)} - y^{(i)} \right)^2
$$
- $m$ is the number of training instances.
- $\mathbf{x}^{(i)}$ and $y^{(i)}$ are the feature vector and target of the $i^{\text{th}}$ instance.

> [!note] Assumptions
> Linear regression trained using MSE assumes the data is **purely linear** and that the noise is **Gaussian** (e.g. outliers are exponentially rare). The more wrong this assumption is, the more biased the model will be.

To find the $\boldsymbol{\theta}$ that minimizes the MSE, there's a **closed-form solution** — a formula that gives the result directly, without using [[Gradient Descent]]:
$$
\hat{\boldsymbol{\theta}} = \left( \mathbf{X}^\intercal \mathbf{X} \right)^{-1} \mathbf{X}^\intercal \mathbf{y}
$$
See [[Normal Equation]] for the full derivation.

>[!tip] Training loss vs. performance metric
>Learning algorithms often **optimize a different loss function during training than the metric used to evaluate the final model** — because the loss is easier to optimize, or carries extra terms needed only during training (e.g. for regularization).
>- A good **performance metric** is as close as possible to the final business objective.
>- A good **training loss** is easy to optimize and strongly correlated with that metric.
>
>For example, classifiers are often trained with the #log-loss but evaluated with #precision / #recall — the log loss is easy to minimize, and doing so usually improves precision/recall.

## Using Scikit-Learn
Performing linear regression with Scikit-Learn is straightforward — it computes the parameters for you and exposes them as `intercept_` (the bias term $\theta_0$) and `coef_` (the feature weights):

```python
import numpy as np

# generate synthetic dataset
rng = np.random.default_rng(seed=42)
m = 200 # number of instances
X = 2 * rng.random((m, 1)) # column vector
y = 4 + 3 * X + rng.standard_normal((m, 1)) # column vector
```

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.intercept_, lin_reg.coef_ # (array([3.69084138]), array([[3.32960458]]))

X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)
# array([[ 3.69084138],
#        [10.35005055]])
```

```python
import matplotlib.pyplot as plt

# plot the fitted line
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
[...] # beautify the figure: add labels, axis, grid, and legend
plt.show()
```

![[4-2.png]]