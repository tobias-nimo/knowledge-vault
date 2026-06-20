# Regularized Linear Models
A good way to reduce #overfitting is to **regularize the model**, i.e. constrain it: the fewer degrees of freedom it has, the harder it is to overfit.

In [[Linear Regression|linear models]], regularization works by **penalizing large weights**, encouraging the model to keep them small. This often improves generalization, **making linear models more stable and accurate** on unseen data.
  
There are several ways to apply this penalty, each constraining the model's weights differently.

> [!warning] Scale your features first!
> #normalization #standardization 
> 
> Regularized models are **sensitive to the scale of the input features**, so scale the data (e.g. with `StandardScaler`) before training. This is true of most regularized models.

## Ridge Regression
#ridge-regression adds a regularization term $\frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2$ to the [[Linear Regression|MSE]]. This forces the algorithm to not only fit the data but also **keep the weights as small as possible**, making the model less flexible so it can't stretch itself to chase every data point.
$$
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2
$$
- $\alpha$ is the **regularization hyperparameter**.
	- If $\alpha = 0$, ridge is just plain linear regression.
	- As $\alpha$ increases, the weights are pushed closer to zero.
	- If $\alpha$ is very large, all weights collapse toward zero.
- The sum starts at $i = 1$, so the **bias term $\theta_0$ is not regularized**.

If $\mathbf{w}$ denotes the vector of feature weights ($\theta_1$ to $\theta_n$), the regularization term can be written as $\frac{\alpha}{m}\lVert \mathbf{w} \rVert_2^2$, the squared [[Ways to Measure Distance|L2 norm]] of $\mathbf{w}$, which is why ridge regression is also known as **L2 regularization**.

> [!note] Train vs. evaluate
> The regularization term is added **only during training**. Once trained, evaluate the model with the unregularized #MSE (or #RMSE).

First, we generate a small synthetic dataset:

```python
import numpy as np

# generate synthetic dataset
rng = np.random.default_rng(seed=42)
m = 20  # number of instances

X = 3 * rng.random((m, 1))
y = 1 + 0.5 * X + rng.standard_normal((m, 1)) / 1.5
```

As with linear regression, you can solve ridge with a **closed-form equation**:

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky") 
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])  # array([1.84414523])
```

 Or with **gradient descent** ( #SGD ):
 
```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
	penalty="l2", # adds alpha times the square of the L2 norm to the MSE — just like ridge, except there's no division by m...
	alpha=0.1 / m, # so we pass 0.1 / m here, to match Ridge(alpha=0.1)
	tol=None,
	max_iter=1000,
	eta0=0.01,
	random_state=42
)

sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
sgd_reg.predict([[1.5]])   # array([1.83659707])
```

Now, let's train a [[Polynomial Regression|polynomial]] ridge model:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
	PolynomialFeatures(degree=10, include_bias=False),
	StandardScaler(),
	Ridge(alpha=1e-05, solver="cholesky")  # or SGDRegressor
)

model.fit(X, y)
model.predict([[1.5]])
```

![[4-18.png]]
> Linear (left) and polynomial (right) ridge models at various $\alpha$ levels. Increasing $\alpha$ leads to flatter, more reasonable predictions: **less variance but more bias**.

## Lasso Regression
#lasso-regression (*least absolute shrinkage and selection operator*) is another regularized linear regression, but it uses the [[Ways to Measure Distance|L1 norm]] of the weight vector instead of the square of the L2 norm:
$$
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + 2\alpha\sum_{i=1}^{n}\lvert\theta_i\rvert
$$
Note the L1 norm is multiplied by $2\alpha$, whereas ridge's L2 term used $\alpha/m$. These factors are chosen so the **optimal $\alpha$ is independent of the training set size**; different norms lead to different factors.

The lasso cost is **not differentiable at $\theta_i = 0$**, but gradient descent still works if you use a **subgradient vector** there:
$$
g(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}}\,\text{MSE}(\boldsymbol{\theta}) + 2\alpha
\begin{bmatrix}
\text{sign}(\theta_1) \\
\text{sign}(\theta_2) \\
\vdots \\
\text{sign}(\theta_n)
\end{bmatrix}
\quad\text{where}\quad
\text{sign}(\theta_i) =
\begin{cases}
-1 & \text{if } \theta_i < 0 \\
\;\;\,0 & \text{if } \theta_i = 0 \\
+1 & \text{if } \theta_i > 0
\end{cases}
$$
The key property: lasso tends to **eliminate the weights of the least important features**, setting them to exactly zero. In the polynomial plot below (with $\alpha = 0.01$) the curve looks roughly cubic — all the high-degree weights are zero. In other words, lasso **automatically performs feature selection and outputs a sparse model**. The trade-off: push $\alpha$ too high and the model gets very sparse but its performance plummets.

Let's implement it with the **closed-form equation**

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])  # array([1.87550211])
```

Now, with #SGD:

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
	penalty="l1", # just like lasso regularization term
	alpha=0.1,
	tol=None,
	max_iter=1000,
	eta0=0.01, # initial learning rate  
	learning_rate="adaptive", # helps Lasso converge by reducing the step size over time (recommended)
	random_state=42
)

sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```

![[4-19.png]]
> Linear (left) and polynomial (right) lasso models at various $\alpha$ levels.

## Elastic Net Regression
**Elastic net** is a middle ground between ridge and lasso: the regularization term is a **weighted sum of both**, controlled by the mix ratio $r$.
$$
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r\left(2\alpha\sum_{i=1}^{n}\lvert\theta_i\rvert\right) + (1 - r)\left(\frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2\right)
$$
At $r = 0$ it's ridge; at $r = 1$ it's lasso.

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5) # l1_ratio is the mix ratio r
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])  # array([1.8645014])
```

> [!tip] Which one should you use?
> - It's almost always preferable to have **at least a little regularization**, so generally **avoid plain linear regression**.
> - **Ridge** is a good default.
> - If you suspect **only a few features are useful**, prefer **lasso** or **elastic net**, since they drive useless weights to zero.
> - Prefer **elastic net over lasso** in general: lasso can behave erratically when there are **more features than training instances**, or when several features are **strongly correlated**.

> [!tip] Efficient CV variants  
> Several regularized linear models have specialized cross-validation estimators that **automatically tune their hyperparameters using cross-validation**. These are roughly equivalent to `GridSearchCV` but are optimized for their specific models, making them much faster. Examples include `RidgeCV`, `LassoCV`, and `ElasticNetCV`.
