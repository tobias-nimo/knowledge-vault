# Decision Trees for Regression
## Using Scikit-Learn
Let's fit a `DecisionTreeRegressor` on a noisy quadratic dataset:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(seed=42)
X_quad = rng.random((200, 1)) - 0.5 # single input feature
y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)

[...] # visualize the tree graph
[...] # plot the fitted model
```
![[5-4.png]]
> A regression tree: each node predicts a value instead of a class.

To predict for $x_1 = 0.2$: the root asks $x_1 \le 0.343$ (yes → left), then $x_1 \le -0.302$ (no → right), reaching a leaf that predicts `value=0.038`. This is the **average target value** of the 133 training instances in that leaf, with an MSE of 0.002 over them.

Without regularization, regression and classification trees are prone to overfitting the data. Let's inspect the model predictions under different hyperparameters:

![[5-5.png]]
> Left: `max_depth=2` possibly underfitting. Right: `max_depth=3` seems better.

![[5-6.png]]
> Left: default hyperparameters badly overfit. Right: setting `min_samples_leaf=10` gives a far more reasonable model.

Note that each region's prediction is the **average target value** of the instances it contains.
## The CART Training Algorithm
CART works the same way as in [[Classification Tasks|classification]], but instead of minimizing impurity it now **minimizes the MSE**:
$$
J(k, t_k) = \frac{m_\text{left}}{m} \text{MSE}_\text{left} + \frac{m_\text{right}}{m} \text{MSE}_\text{right}
$$
where
$$
\text{MSE}_\text{node} = \frac{\sum_{i \in \text{node}} \big(\hat{y}_\text{node} - y^{(i)}\big)^2}{m_\text{node}}
\qquad
\hat{y}_\text{node} = \frac{\sum_{i \in \text{node}} y^{(i)}}{m_\text{node}}
$$
