# Normal Equation
The #normal-equation is the **closed-form solution** to [[Linear Regression]] — a formula that gives the $\boldsymbol{\theta}$ minimizing the MSE directly, without iterating:
$$
\hat{\boldsymbol{\theta}} = \left( \mathbf{X}^\intercal \mathbf{X} \right)^{-1} \mathbf{X}^\intercal \mathbf{y}
$$
- $\hat{\boldsymbol{\theta}}$ is the value of $\boldsymbol{\theta}$ that minimizes the cost function.
- $\mathbf{X}$ is the matrix of all feature vectors, with one row per instance (and the bias column $x_0 = 1$).
- $\mathbf{y}$ is the vector of target values $y^{(1)}$ to $y^{(m)}$.

## Where it comes from
$$
\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^\intercal \mathbf{x}^{(i)} - y^{(i)} \right)^2
$$
Stacking all $m$ instances, the predictions are $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$, so the MSE is just the squared length of the **residual vector** $\mathbf{X}\boldsymbol{\theta} - \mathbf{y}$:
$$
\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \lVert \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \rVert^2 = \frac{1}{m} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^\intercal (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})
$$

This is a **convex** function of $\boldsymbol{\theta}$, so its single minimum is wherever the gradient is zero. Expanding the product gives three terms:
$$
\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \left( \boldsymbol{\theta}^\intercal \mathbf{X}^\intercal \mathbf{X} \boldsymbol{\theta} - 2\,\mathbf{y}^\intercal \mathbf{X} \boldsymbol{\theta} + \mathbf{y}^\intercal \mathbf{y} \right)
$$

Taking the gradient with respect to $\boldsymbol{\theta}$ (using $\nabla (\boldsymbol{\theta}^\intercal \mathbf{A} \boldsymbol{\theta}) = 2\mathbf{A}\boldsymbol{\theta}$ for symmetric $\mathbf{A} = \mathbf{X}^\intercal \mathbf{X}$, and $\nabla (\mathbf{b}^\intercal \boldsymbol{\theta}) = \mathbf{b}$):
$$
\nabla_{\boldsymbol{\theta}} \, \text{MSE}(\boldsymbol{\theta}) = \frac{2}{m} \left( \mathbf{X}^\intercal \mathbf{X} \boldsymbol{\theta} - \mathbf{X}^\intercal \mathbf{y} \right)
$$

Setting this to zero gives the **normal equations** $\mathbf{X}^\intercal \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^\intercal \mathbf{y}$, and solving for $\boldsymbol{\theta}$ (assuming $\mathbf{X}^\intercal \mathbf{X}$ is invertible) recovers the formula above:
$$
\hat{\boldsymbol{\theta}} = \left( \mathbf{X}^\intercal \mathbf{X} \right)^{-1} \mathbf{X}^\intercal \mathbf{y}
$$

>[!note] Why "normal"?
>At the optimum the residual $\mathbf{X}\hat{\boldsymbol{\theta}} - \mathbf{y}$ is **orthogonal** (normal) to the column space of $\mathbf{X}$ — exactly the condition $\mathbf{X}^\intercal (\mathbf{X}\hat{\boldsymbol{\theta}} - \mathbf{y}) = \mathbf{0}$ states.
