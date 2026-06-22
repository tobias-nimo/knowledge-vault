# Logistic Regression
#logistic-regression is commonly used to **estimate the probability that an instance belongs to a particular class**. If the estimated probability exceeds a threshold (typically 50%), the model predicts the instance belongs to the positive class otherwise it predicts the negative class. This makes it a **binary classifier**.

Just like a [[Linear Regression|linear regression]] model, logistic regression computes a weighted sum of the input features (plus a bias term). But instead of outputting the result directly, it outputs the **logistic** of that result:
$$
\hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma\!\left(\boldsymbol{\theta}^{\!\top}\mathbf{x}\right)
$$
The logistic $\sigma(\cdot)$ is a **sigmoid function** (i.e. S-shaped) that outputs a number between 0 and 1:
$$
\sigma(t) = \frac{1}{1 + \exp(-t)}
$$
![[4-22.png]]

Once the model has estimated the probability $\hat{p}$ that an instance belongs to the positive class, the prediction $\hat{y}$ is straightforward:
$$
\hat{y} =
\begin{cases}
0 & \text{if } \hat{p} < 0.5 \\
1 & \text{if } \hat{p} \ge 0.5
\end{cases}
$$
Notice that $\sigma(t) < 0.5$ when $t < 0$ and $\sigma(t) \ge 0.5$ when $t \ge 0$, so with the default 50% threshold the model predicts **1 if $\boldsymbol{\theta}^{\!\top}\mathbf{x}$ is positive and 0 if it is negative**.
## Training: Minimizing the Log Loss
The objective of training is to set $\boldsymbol{\theta}$ so the model estimates high probabilities for positive instances ($y = 1$) and low probabilities for negative instances ($y = 0$). For a single training instance, this is captured by:
$$
c(\boldsymbol{\theta}) =
\begin{cases}
-\log(\hat{p}) & \text{if } y = 1 \\
-\log(1 - \hat{p}) & \text{if } y = 0
\end{cases}
$$
This makes sense because $-\log(t)$ grows very large as $t \to 0$, so the cost is large if the model estimates a probability close to 0 for a positive instance (or close to 1 for a negative one).

The cost over the whole training set is the average cost across all instances, written in a single expression called the #log-loss:
$$
J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\!\big(\hat{p}^{(i)}\big) + \big(1 - y^{(i)}\big) \log\!\big(1 - \hat{p}^{(i)}\big) \right]
$$

> [!note] Assumptions
> Log loss assumes the instances of each class follow a **Gaussian distribution around their class mean**. The more wrong this assumption is, the more biased the model will be.
 
There's no closed-form equation to minimize this cost. But the cost function is **convex**, so [[Gradient Descent|gradient descent]] is guaranteed to find the global minimum (given a small enough learning rate and enough time). The partial derivatives and gradient vector are:
$$
\frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \sigma\!\big(\boldsymbol{\theta}^{\!\top}\mathbf{x}^{(i)}\big) - y^{(i)} \right) x_j^{(i)}
$$
$$
\nabla_{\boldsymbol{\theta}}\,J(\boldsymbol{\theta}) =
\begin{bmatrix}
\frac{\partial}{\partial \theta_0} J(\boldsymbol{\theta}) \\[4pt]
\frac{\partial}{\partial \theta_1} J(\boldsymbol{\theta}) \\[4pt]
\vdots \\[4pt]
\frac{\partial}{\partial \theta_n} J(\boldsymbol{\theta})
\end{bmatrix}
$$
From here, plug the gradient vector into batch GD (or feed one instance at a time for SGD, a mini-batch for mini-batch GD).
## Using Scikit-Learn
Let's use the [[Iris|Iris dataset]] to build a classifier to **detect *Iris virginica* based only on the petal width** feature:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

The `predict()` method just returns the most likely class, while `predict_proba` returns the estimated probability:

```python
log_reg.predict([[1.7], [1.5]])  # array([ True, False])

log_reg.predict_proba([[1.7], [1.5]])  
# array([[0.4528, 0.5471], -> 45% not-virginica and 55% virginica
#       [0.6403, 0.3596]]) -> 64% not-virginica and 36% virginica
```

Let's plot at the **estimated probabilities** and the **decision boundary**:

```python
import numpy as np

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # reshape to a column vector
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0] # np.float64(1.6516)

plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", label="Decision boundary")
plt.show()
```
![[4-24.png]]

The decision boundary is at around 1.6 cm, where both probabilities equal 50%. Above it the model predicts virginica, below it predicts not — even when it isn't very confident around ~1 and ~2 cm.

Now, let's train a new model to **detect *Iris virginica* but using petal width and petal length**:

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)
```

With two features, the model's 50%-probability decision boundary is a straight line — logistic regression always produces a **linear decision boundary**.

```python
# for the contour plot
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]  # one instance per point on the figure
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)

# for the decision boundary
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])

plt.figure(figsize=(10, 4))
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "bs")
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "g^")
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
[...]
plt.show()
```
![[4-25.png]]
> The dashed line marks the 50% decision boundary where $\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$; each parallel line marks a fixed probability, from 15% to 90%.

> [!tip] Regularization
> 
> Like other [[Regularization|linear models]], logistic regression can be regularized with $\ell_1$ or $\ell_2$ penalties — Scikit-Learn adds an $\ell_2$ penalty by default.
> 
> The hyperparameter controlling regularization strength in Scikit-Learn's `LogisticRegression` is not `alpha` (as in other linear models) but its **inverse, `C`**. The **higher** `C` is, the **less** the model is regularized.