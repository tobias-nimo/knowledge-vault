# Stochastic Gradient Descent
Where [[Batch GD|batch gradient descent]] averages the gradient over all $m$ instances each step, #SGD picks **one random instance** $i$ and uses its gradient alone:
$$
\nabla_{\boldsymbol{\theta}}\,\text{MSE}^{(i)}(\boldsymbol{\theta}) = 2\left(\boldsymbol{\theta}^\intercal \mathbf{x}^{(i)} - y^{(i)}\right)\mathbf{x}^{(i)}
$$
Note there's no $\tfrac{1}{m}$ averaging — it's a single, noisy estimate of the full-batch gradient. Each step is very fast, needs little memory, and can train on huge sets (it works as an out-of-core algorithm).

The price is irregularity: instead of gently decreasing, the cost **bounces up and down, decreasing only on average**. It ends up close to the minimum but keeps bouncing, never settling — so the final parameters are good, but not optimal.
![[4-9.png]]

This randomness has an upside: on irregular cost functions it helps the algorithm **jump out of local minima**, giving it a better shot at the global minimum than batch GD. The dilemma — randomness escapes local optima but prevents settling — is resolved by gradually reducing the learning rate. Steps start large (quick progress, escape local minima) then shrink, letting it settle. 

The function setting the rate each iteration is the #learning-schedule :
- Reduce the rate **too quickly** → you may get stuck in a local minimum or freeze halfway.
- Reduce it **too slowly** → you bounce around the minimum for a long time and may end up suboptimal if you stop early.

A quick implementation:

```python
from my_datasets import X_b, y, m  # defined in ./batch-gd

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

By convention each round of $m$ iterations is an #epoch. Where [[Batch GD]] iterated 1,000 times over the whole set, this reaches a good solution in just 50 passes.
![[4-10.png]]

> [!warning] **Training instances must be IID**  
> SGD assumes training instances are **independent and identically distributed** ( #IID ). For example, if instances are ordered (e.g., by label), SGD may optimize for one subset at a time and fail to converge near the global optimum. 
> 
> Note that random sampling can draw some instances multiple times while skipping others in an epoch, while sequentially iterating through a shuffled dataset avoids this but is usually no better in practice.

You can use #sklearn `SGDRegressor`, which defaults to optimizing the MSE, to perform **linear regression using SGD**:

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
	max_iter=1000,        # max 1000 epochs...
	tol=1e-5,             # or stop if loss drops < 1e-5 (tol)...
	n_iter_no_change=100, # and over 100 epochs;
	penalty=None,         # no regularization
	eta0=0.01,            # initial learning rate 0.01 
	random_state=42
)

sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets

sgd_reg.intercept_, sgd_reg.coef_  # (array([3.68899733]), array([3.33054574]))
```

