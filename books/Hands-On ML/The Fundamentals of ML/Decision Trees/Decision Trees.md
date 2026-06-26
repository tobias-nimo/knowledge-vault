# Decision Trees
A #decision-tree is a versatile algorithm that can perform both classification and regression, and is powerful enough to fit complex datasets. Check these notes:
- [[Classification Tasks]]
- [[Regression Tasks]]

> [!note] Binary trees only
> Scikit-Learn uses the **CART** algorithm, which produces only **binary trees** — split nodes always have exactly two children (yes/no questions). Other algorithms such as **ID3** can produce nodes with more than two children.

## Regularization Hyperparameters
A decision tree is a **nonparametric model**: not because it has no parameters, but because their *number is not fixed before training*, so the structure is free to stick to the data. A **parametric model** (like a linear model) has a predetermined number of parameters, limiting its freedom (less overfitting, but more risk of underfitting).

So, decision trees make **very few assumptions** about the data. Left unconstrained, the tree adapts itself closely to the training data — most likely #overfitting it.

> [!warning] Regularization is a must!
> Decision trees are prone to overfitting the data — default hyperparameters badly overfit.

To regularize, restrict the tree's freedom during training. The key #regularization knobs in Scikit-Learn:
- `max_depth` — maximum depth (default `None` = unlimited). 
- `max_features` — max features evaluated for splitting at each node. Great for high-dimensional data.
- `max_leaf_nodes` — maximum number of leaf nodes.
- `min_samples_split` — minimum samples a node must have before it can be split.
- `min_samples_leaf` — minimum samples a leaf must have. A good idea especially for small datasets.
- `min_weight_fraction_leaf` — same as `min_samples_leaf`, but as a fraction of the total weighted instances.
- `min_impurity_decrease` — only split if it reduces impurity by at least this much.
- `ccp_alpha` — controls minimal **cost-complexity pruning** (prunes subtrees that don't reduce impurity enough relative to their leaf count). Default `0` (no pruning).

Increasing `min_*` hyperparameters or `ccp_alpha`, or decreasing `max_*` hyperparameters, all **regularize** the model.

> [!tip] 
> Usually `max_features` is the best default knob: effective and keeps the tree small and interpretable.

Testing regularization on the [[Moons|moons dataset]] (two interleaving crescents):

```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier(random_state=42) # unregularized
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)

tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
```
![[5-3.png]]
> Decision boundaries of an unregularized tree (left, overfitting) and a regularized tree with `min_samples_leaf=5` (right, generalizes better).

> [!tip] No data preparation needed
> One of the great qualities of decision trees is that they require **very little data preparation**:
>  
>  - Scale does not affect their training — no feature scaling or centering needed.
>  - `DecisionTreeClassifier` and `DecisionTreeRegressor` both support #missing-values natively — no imputer needed.

Evaluating both on a fresh test set (different seed) confirms it:

```python
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)

tree_clf1.score(X_moons_test, y_moons_test)  # 0.85 (unregularized)
tree_clf2.score(X_moons_test, y_moons_test)  # 0.92 (regularized)
```

## Limitations
### Sensitivity to Axis Orientation
Trees love **orthogonal decision boundaries** (every split is perpendicular to an axis), which makes them **sensitive to the data's orientation**. 

For example, a linearly separable dataset is split easily as-is, but after rotating it 45° the boundary becomes unnecessarily convoluted and likely won't generalize.

![[5-7.png]]
> [[Iris|Iris dataset]] (petal length and width only): before (left) and after a 45° rotation (right). Both trees fit perfectly, but the rotated one generalizes poorly.

One fix: **scale the data, then apply PCA**, which rotates the features to reduce their correlation — often (not always) making things easier for trees.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)

tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)

[...] # plot the decision boundaries
```
![[5-8.png]]
> After #standardization  and #PCA rotation, the tree fits the iris dataset well using essentially a single feature $z_1$.
### High Variance
The main issue with trees is high variance: **small changes to the hyperparameters or data can produce very different models**. Because Scikit-Learn's training is stochastic (it randomly selects the features to evaluate at each node), even retraining on the *exact same data* can yield a very different tree — unless you set `random_state`.

![[5-9.png]]
> Retraining the same model on the same data may produce a very different tree.

Luckily, averaging predictions over many trees reduces variance significantly. Such an ensemble of trees is called a [[Random Forests]] — one of the most powerful models available.