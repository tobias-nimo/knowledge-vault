# Random Forests
A #random-forest is an ensemble of [[Decision Trees|decision trees]], generally trained via [[Bagging and Pasting|bagging]], typically with `max_samples` set to the full size of the training set.

A random forest adds extra randomness compared to a plain #decision-tree: when splitting a node, instead of searching for the very best feature, each tree **searches for the best feature among a random subset of features** (by default $\sqrt{n}$, where $n$ is the total number of features). 

This increases tree diversity (i.e., makes the trees less correlated), slightly increasing bias while **reducing variance**, generally yielding a better model overall.

For example, this `BaggingClassifier` is roughly equivalent to a random forest with 500 trees, each capped at 16 leaf nodes, using all CPU cores:

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500,
    n_jobs=-1,
    random_state=42
)
```

Instead of building a `BaggingClassifier` around a `DecisionTreeClassifier`, use the dedicated `RandomForestClassifier` (or `RandomForestRegressor` for regression) — it's more convenient and optimized for decision trees. 

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
                                 
#rnd_clf.fit(X_train, y_train)
#y_pred_rf = rnd_clf.predict(X_test)
```

> [!note] Hyperparameters
> With few exceptions, a `RandomForestClassifier` exposes all the hyperparameters of a `DecisionTreeClassifier` (to control [[Classification Tasks|how each tree grows]]) along with those of `BaggingClassifier` (to control the [[Bagging and Pasting|ensemble]] itself).

> [!tip] No data preparation needed
> Like decision trees, forests require **very little data preparation**:
>  
>  - Scale does not affect their training — no feature scaling or centering needed.
>  - `RandomForestClassifier` and `RandomForestRegressor` both support #missing-values natively — no imputer needed.

## Extra-Trees
You can push the randomness even further. Like random forests, #extra-trees consider only a random subset of features at each node. However, instead of searching for the **optimal threshold** for each candidate feature, they assign one **random threshold** to each of them and then **select the best split among these candidates**.

Set `splitter="random"` on a `DecisionTreeClassifier` to do this:

```python
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(
    splitter="random",
    max_features="sqrt",
    random_state=42
)
```

A forest of such trees is an *extremely randomized trees* ensemble. This increases tree diversity even further, trading **more bias for lower variance**, so extra-trees can outperform regular random forests when the latter tend to overfit, especially on noisy or high-dimensional datasets. 

They're also **much faster to train** because searching for the optimal threshold is one of the most expensive parts of growing a decision tree, and extra-trees avoid that search.
## Feature Importance
Another strength of random forests is that they make it easy to **estimate the relative importance of each feature**.

Scikit-Learn computes feature importance by measuring how much each feature reduces impurity on average across all trees in the forest. More precisely, each split is weighted by the number of training samples that reach the corresponding node.

The scores are computed automatically after training and normalized so they sum to 1. They are available through the `feature_importances_` attribute. For example, on the [[Iris|Iris dataset]], petal length and width dominate:

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)

for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)

# 0.11 sepal length (cm)
# 0.02 sepal width (cm)
# 0.44 petal length (cm)
# 0.42 petal width (cm)
```

Random forests are very handy for getting a quick understanding of which features actually matter, especially when you need to perform #feature-selection.