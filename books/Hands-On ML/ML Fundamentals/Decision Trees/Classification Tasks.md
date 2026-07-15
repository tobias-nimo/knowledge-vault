# Decision Trees for Classification

## Using Scikit-Learn
Let's train a `DecisionTreeClassifier` on the [[Iris|Iris dataset]], using only petal length and width:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

[...] # plot the decision boundaries
```

The trained model produces the following **decision boundaries** (always orthogonal to an axis):

![[5-2.png]]
> The thick vertical line is the root split; the dashed line is the depth-1 right split. Because `max_depth=2`, the tree stops here — bumping it to `3` would let each depth-2 node add one more boundary (the two vertical dotted lines).

You can **visualize the trained tree** like this:

```python
from sklearn.tree import export_graphviz
from graphviz import Source

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
Source.from_file("iris_tree.dot")
```

![[5-1.png]]
> The iris decision tree. To classify a flower, start at the **root node** (depth 0) and follow the questions down until you reach a **leaf node**, where the `class` attribute indicates the predicted class.

Each node carries a few useful attributes:
- `samples` — how many training instances reach this node.
- `value` — how many of those belong to each class.
- `gini` — the node's **Gini impurity**. 

You can compute the Gini impurity $G_i$ of the $i$-th node as:
$$
G_i = 1 - \sum_{k=1}^{n} p_{i,k}^{2}
$$
- A node is *pure* (`gini=0`) when all its instances belong to a single class.
- $p_{i,k}$ is the ratio of class-$k$ instances among the training instances in the $i$-th node.

For example, the depth-2 left node has impurity $1 - (0/54)^2 - (49/54)^2 - (5/54)^2 \approx 0.168$.

> [!note] White box model
> Decision trees are a classic [[ML Explainability|white box model]]: their decisions are intuitive and easy to interpret, unlike the black box random forests and neural networks.

A tree can estimate the **probability that an instance belongs to class $k$**: it traverses to the instance's leaf node and returns the proportion of class-$k$ training instances in that leaf. For a flower with petals 5 cm long and 1.5 cm wide, the leaf is the depth-2 left node, giving 0% setosa (0/54), 90.7% versicolor (49/54), and 9.3% virginica (5/54):

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)  # array([[0., 0.907, 0.093]])
tree_clf.predict([[5, 1.5]])                 # array([1])
```

> [!warning]
> The estimated probabilities are **identical anywhere in the same leaf's region**. A flower with petals 6 cm × 1.5 cm gets the same probabilities, even though it intuitively looks much more like virginica.

## The CART Training Algorithm
Scikit-Learn uses the **Classification and Regression Tree (CART)** algorithm to "grow" trees. It splits the training set in two using a single feature $k$ and threshold $t_k$ (e.g. "petal length ≤ 2.45 cm"), choosing the pair $(k, t_k)$ that produces the purest subsets, weighted by their size, by minimizing this cost function:
$$
J(k, t_k) = \frac{m_\text{left}}{m} G_\text{left} + \frac{m_\text{right}}{m} G_\text{right}
$$
- $G_\text{left/right}$ is the impurity of the left/right subset.
- $m_\text{left/right}$ is the number of instances in the left/right subset.
- $m = m_\text{left} + m_\text{right}$.

It then splits each subset with the same logic, recursively, stopping when it reaches `max_depth`, can't find a split that reduces impurity, or hits another stopping condition (`min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, …).

> [!note] A greedy algorithm
> CART is **greedy**: it picks the best split at the top level, then repeats at each level below, without checking whether a split leads to the lowest possible impurity several levels down. This usually yields a *reasonably good* but not optimal solution.

