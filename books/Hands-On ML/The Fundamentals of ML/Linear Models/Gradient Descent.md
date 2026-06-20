# Gradient Descent
Training a model is a search for the parameter combination that minimizes the cost over the training set — **a search in the model's parameter space**. The more parameters, the more dimensions, and the harder the search.

#gradient-descent is a **generic optimization algorithm** that tweaks parameters iteratively to minimize a cost function.

The intuition: imagine you're lost in the mountains in dense fog and can only feel the slope under your feet. To reach the valley quickly, you head downhill in the direction of the steepest slope. Gradient descent does exactly this — it measures the local gradient of the cost function with regard to the parameter vector $\boldsymbol{\theta}$ and moves in the direction of descending gradient. Once the gradient is zero, you've reached a minimum.

In practice you start by filling $\boldsymbol{\theta}$ with random values (*random initialization*), then improve it one baby step at a time, each step trying to decrease the cost (e.g. the [[Linear Regression|MSE]]), until the algorithm converges.
![[4-3.png]]

> [!note] About #MSE 
> Fortunately, the **MSE cost function for [[Linear Regression|linear regression]] is convex**: any line segment joining two points on the curve never dips below it, so there are no local minima — just one global minimum. It's also continuous with a slope that never changes abruptly. Together these mean gradient descent is **guaranteed to approach the global minimum arbitrarily closely**, given enough time and a learning rate that isn't too high.

## Common Pitfalls
### Learning Rate
The step size is proportional to the slope and controlled by the #learning-rate hyperparameter, so it gradually shrinks as the cost approaches the minimum.

![[4-4.png]]
> **Learning rate is too small** → many iterations are needed to converge, which takes a long time.

![[4-5.png]]
> **Learning rate is too high** → the algorithm can **diverge**, with larger and larger values, failing to find a good solution.

The choice of learning rate has a major impact on how gradient descent behaves. The figure below shows the first 20 steps of the algorithm when training a linear regression model with three different learning rates:
![[4-8.png]]
> too low (left, slow), good (middle, converges in a few epochs), and too high (right, diverges and jumps all over the place).

> [!tip]
> To find a good #learning-rate, use #grid-search — but cap the number of epochs so it can eliminate models that converge too slowly. 
### Local Minima and Plateaus
Not all cost functions are nice regular bowls like MSE— there may be holes, ridges, and plateaus that make convergence hard. For example, see Figure 4-6:
- If random initialization starts you on the left, you converge to a **local minimum**, which is worse than the global minimum.
- If it starts you on the right, crossing the **plateau** takes a very long time, and if you stop too early you never reach the global minimum.

![[4-6.png]]
### Feature scaling
Even a convex bowl like the MSE can be elongated if the **features have very different scales**. 
![[4-7.png]]
> With features on the same scale (left) gradient descent heads straight for the minimum; with very different scales (right) it first moves almost orthogonally to the goal, then crawls down a near-flat valley — much slower.


>[!warning] Scale your features
> #normalization #standardization 
> 
>When using gradient descent, ensure all features have a similar scale (e.g. with Scikit-Learn's `StandardScaler`), or it will take much longer to converge.

## Algorithm Variants
GD variants differ in **how much data they use to compute the gradient at each step**, trading off speed per step, regularity of convergence, and the ability to escape local minima.

### Batch GD
#batch-gradient-descent computes the gradient of the cost function over the **whole training set at every step** — the entire *batch* of data each step (hence the name). It's the most direct form of gradient descent.

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

### Stochastic GD
Where batch gradient descent averages the gradient over all $m$ instances each step, #SGD picks **one random instance** $i$ and uses its gradient alone:
$$
\nabla_{\boldsymbol{\theta}}\,\text{MSE}^{(i)}(\boldsymbol{\theta}) = 2\left(\boldsymbol{\theta}^\intercal \mathbf{x}^{(i)} - y^{(i)}\right)\mathbf{x}^{(i)}
$$
Note there's no $\tfrac{1}{m}$ averaging — it's a single, noisy estimate of the full-batch gradient. Each step is very fast, needs little memory, and can train on huge sets (it works as an out-of-core algorithm).

The price is irregularity: instead of gently decreasing, the cost **bounces up and down, decreasing only on average**. It ends up close to the minimum but keeps bouncing, never settling — so the final parameters are good, but not optimal.
![[4-9.png]]

This randomness has an upside: on irregular cost functions it helps the algorithm **jump out of local minima**, giving it a better shot at the global minimum than batch GD. The dilemma — randomness escapes local optima but prevents settling — is resolved by gradually reducing the learning rate. Steps start large (quick progress, escape local minima) then shrink, letting it settle. 

The function setting the rate each iteration is the #learning-schedule:
- Reduce the rate **too quickly** → you may get stuck in a local minimum or freeze halfway.
- Reduce it **too slowly** → you bounce around the minimum for a long time and may end up suboptimal if you stop early.

A quick implementation:

```python
# X_b, y, m from the Batch GD code above

n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))  # randomly initialized parameters

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = rng.integers(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

theta
# array([[3.69826475],
#        [3.30748311]])
```

By convention each round of $m$ iterations is an #epoch. Where batch GD iterated 1,000 times over the whole set, this reaches a good solution in just 50 passes.
![[4-10.png]]

> [!warning] **Training instances must be IID**  
> SGD assumes training instances are **independent and identically distributed** ( #IID ). For example, if instances are ordered (e.g., by label), SGD may optimize for one subset at a time and fail to converge near the global optimum. 
> 
> Note that random sampling can draw some instances multiple times while skipping others in an epoch, while sequentially iterating through a shuffled dataset avoids this but is usually no better in practice.

You can use #sklearn `SGDRegressor`, which defaults to optimizing the MSE, to perform **linear regression using SGD**:

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
	max_iter=1000,        # max 1000 epochs
	tol=1e-5,
	n_iter_no_change=100,
	penalty=None,         # no regularization
	eta0=0.01,            # initial learning rate 0.01 
	random_state=42
)

sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets

sgd_reg.intercept_, sgd_reg.coef_  # (array([3.68899733]), array([3.33054574]))
```

Training doesn't always run the full `max_iter` epochs. SGD stops early once the loss fails to improve by at least `tol` for `n_iter_no_change` consecutive epochs. Setting `tol=None` disables this check, forcing all `max_iter` epochs.

> [!tip] #early-stopping
> With `SGDRegressor`, set `early_stopping=True` to hold out a `validation_fraction` of the data and monitor the **validation score** instead of the training loss. Training then stops once that score stops improving by at least `tol` for `n_iter_no_change` epochs — i.e. at the bottom of the [[Learning Curves|validation curve]], just before it turns back up. This is a form of #regularization: it prevents the model from overfitting the training data.

### Mini-Batch GD
At each step #mini-batch-gradient-descent computes the gradient on a **small random set of instances** called a *mini-batch*.

Concretely, it computes the gradient over a mini-batch $B$ of $n_b$ instances:
$$
\nabla_{\boldsymbol{\theta}}\,\text{MSE}_B(\boldsymbol{\theta}) = \frac{2}{n_b} \mathbf{X}_B^\intercal \left( \mathbf{X}_B \boldsymbol{\theta} - \mathbf{y}_B \right)
$$
This sits between the two extremes: $n_b = m$ recovers batch GD, $n_b = 1$ recovers SGD.

Its main advantage over SGD is a **performance boost from hardware acceleration** of matrix operations, especially on GPUs. Also, its progress is less erratic than SGD (more so with larger mini-batches), so it ends up walking a bit closer to the minimum — but it may also find it harder to escape local minima on problems that have them.

The next figure shows the paths of all three variants in parameter space: batch GD's path stops at the minimum, while SGD and mini-batch GD keep walking around it (though a good learning schedule would let them settle too).
![[4-11.png]]
