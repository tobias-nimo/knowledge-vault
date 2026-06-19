# Batch Gradient Descent
#batch-gradient-descent computes the gradient of the cost function over the **whole training set at every step** — the entire *batch* of data each step (hence the name). It's the most direct form of #gradient-descent.

To implement it you need to know how much the cost changes if you tweak each parameter $\theta_j$ a little — the **partial derivative** with respect to $\theta_j$. For example, for the [[Linear Regression|MSE]] cost function:
$$
\frac{\partial}{\partial \theta_j} \text{MSE}(\boldsymbol{\theta}) = \frac{2}{m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^\intercal \mathbf{x}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$
Rather than compute these one at a time, the **gradient vector** $\nabla_{\boldsymbol{\theta}}\,\text{MSE}(\boldsymbol{\theta})$ collects them all in one go:
$$
\nabla_{\boldsymbol{\theta}}\,\text{MSE}(\boldsymbol{\theta}) =
\begin{bmatrix}
\frac{\partial}{\partial \theta_0} \text{MSE}(\boldsymbol{\theta}) \\[4pt]
\frac{\partial}{\partial \theta_1} \text{MSE}(\boldsymbol{\theta}) \\[4pt]
\vdots \\[4pt]
\frac{\partial}{\partial \theta_n} \text{MSE}(\boldsymbol{\theta})
\end{bmatrix}
= \frac{2}{m} \mathbf{X}^\intercal (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})
$$
- $m$ is the number of training instances.
- $\mathbf{X}$ is the matrix of all feature vectors (with the bias column $x_0 = 1$).

This formula uses the **full training set $\mathbf{X}$ at every step**, which makes it terribly slow on large training sets — but it scales well with the number of features: training with hundreds of thousands of features is much faster with gradient descent than with the [[Normal Equation]].

The gradient points uphill, so to go downhill you subtract it from $\boldsymbol{\theta}$, scaled by the learning rate $\eta$:
$$
\boldsymbol{\theta}^{\text{(next step)}} = \boldsymbol{\theta} - \eta \, \nabla_{\boldsymbol{\theta}}\,\text{MSE}(\boldsymbol{\theta})
$$
A quick implementation:

```python
import numpy as np
from sklearn.preprocessing import add_dummy_feature

# generate synthetic dataset
rng = np.random.default_rng(seed=42)
m = 200 # number of instances
X = 2 * rng.random((m, 1)) # column vector
y = 4 + 3 * X + rng.standard_normal((m, 1)) # column vector

# add the bias column x0 = 1 to each instance
X_b = add_dummy_feature(X)
```

```python
eta = 0.1         # learning rate
n_epochs = 1000   # epoch: one iteration over the training set
m = len(X_b)      # number of instances

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))  # randomly initialized parameters

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

theta
# array([[3.69084138],
#        [3.32960458]])
```