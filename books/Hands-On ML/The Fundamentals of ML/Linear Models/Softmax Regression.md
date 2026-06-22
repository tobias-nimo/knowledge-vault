# Softmax Regression
#softmax-regression **generalizes [[Logistic Regression|logistic regression]] to multiple classes**.

Given an instance $\mathbf{x}$, the model first computes a **score $s_k(\mathbf{x})$ for each class $k$**, then runs those scores through the **softmax function** to estimate a probability per class. The score is just like the linear regression prediction:
$$
s_k(\mathbf{x}) = \big(\boldsymbol{\theta}^{(k)}\big)^{\!\top}\mathbf{x}
$$
- Each class has its **own parameter vector** $\boldsymbol{\theta}^{(k)}$.
- These vectors are typically stored as the rows of a **parameter matrix** $\boldsymbol{\Theta}$.

Once you have every class's score, the **softmax** estimates the probability $\hat{p}_k$ that the instance belongs to class $k$. It exponentiates every score, then normalizes by the sum of all the exponentials:
$$
\hat{p}_k = \sigma\big(\mathbf{s}(\mathbf{x})\big)_k = \frac{\exp\big(s_k(\mathbf{x})\big)}{\sum_{j=1}^{K} \exp\big(s_j(\mathbf{x})\big)}
$$
- $K$ is the number of classes.
- $\mathbf{s}(\mathbf{x})$ is the vector of scores for instance $\mathbf{x}$.
- $\sigma\big(\mathbf{s}(\mathbf{x})\big)_k$ is the estimated probability that $\mathbf{x}$ belongs to class $k$.

Like logistic regression, the classifier predicts the class with the **highest estimated probability** — which is simply the class with the highest score:
$$
\hat{y} = \underset{k}{\arg\max}\;\sigma\big(\mathbf{s}(\mathbf{x})\big)_k = \underset{k}{\arg\max}\;s_k(\mathbf{x})
$$
> The $\arg\max$ operator returns the value of $k$ that maximizes the estimated probability.

> [!warning] One class at a time
> Softmax regression is **multiclass, not multilabel or multioutput**: it predicts a single class per instance, so use it only with **mutually exclusive classes**.

## Training: Minimizing Cross Entropy
The objective is for the model to estimate a **high probability for the target class** (and low for the others). This is achieved by minimizing the [[Cross Entropy|cross entropy]] cost function, which penalizes the model for estimating a low probability for the target class:
$$
J(\boldsymbol{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log\!\big(\hat{p}_k^{(i)}\big)
$$
- $y_k^{(i)}$ is the **target probability** that instance $i$ belongs to class $k$ — generally 1 or 0.
- When there are just two classes ($K = 2$), this is equivalent to logistic regression's #log-loss.

The gradient of the cost for class $k$ lets you run [[Gradient Descent|gradient descent]] (or any optimizer) to find the $\boldsymbol{\Theta}$ that minimizes the cost:
$$
\nabla_{\boldsymbol{\theta}^{(k)}}\,J(\boldsymbol{\Theta}) = \frac{1}{m} \sum_{i=1}^{m} \big(\hat{p}_k^{(i)} - y_k^{(i)}\big)\,\mathbf{x}^{(i)}
$$
## Using Scikit-Learn
Scikit-Learn's `LogisticRegression` uses **softmax regression automatically** when trained on more than two classes (with the default `solver="lbfgs"`). It applies $\ell_2$ regularization by default, controlled by `C` — decrease `C` to increase regularization. Let's classify the [[Iris|iris]] plants into all **three** classes:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)

X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5, 2]]) # array([2])
softmax_reg.predict_proba([[5, 2]]).round(2) # array([[0. , 0.04, 0.96]])
```

![[4-26.png]]
> Softmax **decision boundaries** between any two classes are linear. The curved lines show the *Iris versicolor* probability.

