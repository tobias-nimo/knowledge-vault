# Boosting
#boosting is an ensemble method that **trains several weak learners sequentially**, each one trying to correct its predecessor. The most popular methods are **adaptive boosting**, **gradient boosting**, and **histogram-based gradient boosting**.

> [!warning] No parallelization
> Because each predictor can only be trained after the previous one has been trained and evaluated, boosting **cannot be parallelized**. As a result, it does not scale as well as [[Bagging and Pasting|bagging or pasting]].

> [!note] Other implementations
> Several optimized gradient boosting libraries exist in the Python ecosystem — [XGBoost](https://github.com/dmlc/xgboost), [CatBoost](https://catboost.ai/), and [LightGBM](https://lightgbm.readthedocs.io/en/stable/). Their APIs are very similar to Scikit-Learn's and they add features like GPU acceleration.
## AdaBoost
The idea behind #adaboost (*adaptive boosting*) is to **pay more attention to the training instances that the predecessor underfit**, so new predictors focus more and more on the hard cases.

When training an AdaBoost classifier, the algorithm:
1. Trains a base classifier and makes predictions on the training set.
2. Increases the relative weight of the misclassified instances.
3. Trains a second classifier using the updated weights, makes predictions, updates the weights again, and so on.

Once trained, the ensemble predicts much like bagging or pasting, except **predictors have different weights** depending on their overall accuracy on the weighted training set.

![[6-7.png]]
> AdaBoost sequential training with instance weight updates.

Let's take a closer look at the **AdaBoost algorithm**:

1. Each **instance weight** $w^{(i)}$ is initially set to $1/m$, so they sum to 1. 
2. A first predictor is trained, and its **weighted error rate** $r_1$ is computed on the training set.
	$$r_j = \displaystyle\sum_{\substack{i=1 \\ \hat{y}_j^{(i)} \neq y^{(i)}}}^{m} w^{(i)}$$
	Where $\hat{y}_j^{(i)}$ is the $j$-th predictor's prediction for the $i$-th instance.
3. The **predictor weight** $\alpha_j$ is then computed. The more accurate the predictor, the higher its weight. A predictor guessing randomly gets a weight near zero; one that is most often wrong gets a negative weight.
	$$\alpha_j = \eta \log \frac{1 - r_j}{r_j}$$
	Where $\eta$ is the learning rate (defaults to 1).
4. The instance weights are then updated by **boosting the weights of the misclassified instances** — this encourages the next predictor to pay more attention to them.
	$$w^{(i)} \leftarrow \begin{cases} w^{(i)} & \text{if } \hat{y}_j^{(i)} = y^{(i)} \\ w^{(i)} \exp(\alpha_j) & \text{if } \hat{y}_j^{(i)} \neq y^{(i)} \end{cases}$$
	Afterwards the weights are normalized to sum up to 1 (divided by $\sum_{i=1}^{m} w^{(i)}$).
5. A new predictor is trained on the updated weights and the **process repeats** until the desired number of predictors is reached, or a perfect predictor is found.

To make predictions, AdaBoost computes the predictions of all predictors and weighs them by $\alpha_j$. The predicted class is the one that receives the **majority of weighted votes**:
$$\hat{y}(\mathbf{x}) = \underset{k}{\text{argmax}} \sum_{\substack{j=1 \\ \hat{y}_j(\mathbf{x}) = k}}^{N} \alpha_j$$
Where $N$ is the number of predictors.
### In Scikit-Learn
Scikit-Learn uses a multiclass version of AdaBoost called **SAMME** (*Stagewise Additive Modeling using a Multiclass Exponential loss function*). With just two classes, SAMME is equivalent to AdaBoost.

The following trains an AdaBoost classifier based on 30 **decision stumps** — decision trees with `max_depth=1` (a single decision node plus two leaf nodes), the default base estimator:

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), # decision stump
    n_estimators=30,
    learning_rate=0.5,
    random_state=42,
    algorithm="SAMME"
)

#ada_clf.fit(X_train, y_train)
```

> [!tip] Overfitting
> If your AdaBoost ensemble is overfitting, try **reducing the number of estimators** or **regularizing the base estimator** more strongly.

## Gradient Boosting
#gradient-boosting sequentially adds predictors, each trained to correct its predecessor by **fitting the residual errors** left by the previous predictor.

Using decision trees as base predictors gives *gradient boosted regression trees* ( #GBRT ). Let's walk through it manually on a noisy quadratic dataset:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

m = 100
rng = np.random.default_rng(seed=42)
X = rng.random((m, 1)) - 0.5
noise = 0.05 * rng.standard_normal(m)
y = 3 * X[:, 0] ** 2 + noise  # y = 3x^2 + Gaussian noise

# Train a first weak learner on the data
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# Train a second tree on the residual errors of the first
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

# And a third on the residual errors of the second
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

The ensemble predicts a new instance simply by **summing the predictions of all trees**:

```python
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
# array([0.57356534, 0.0405142 , 0.66914249])
```

![[6-9.png]]
> Left: predictions of the three individual trees (each trained on the previous one's residuals). 
> Right: ensemble's predictions, which gradually get better as trees are added.

### In Scikit-Learn
Rather than do this by hand, use `GradientBoostingRegressor` (or `GradientBoostingClassifier`). This builds the same ensemble as above:

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(
	max_depth=2,
	n_estimators=3,
	learning_rate=1.0, # default 1.0; scales the contribution of each tree
	random_state=42
)

gbrt.fit(X, y)
```

> [!note] Hyperparameters
> Like `RandomForestRegressor`, it has [[Random Forests|tree-growth hyperparameters]] (`max_depth`, `min_samples_leaf`, …) plus [[Bagging and Pasting|ensemble hyperparameters]] (`n_estimators`, …).

The `GradientBoostingRegressor` class also supports **other important hyperparameters**:
- `learning_rate` scales the contribution of each tree. A lower learning rate (say `0.05`) requires more trees (say 500) to fit the training data, but often improves generalization. This regularization technique is called #shrinkage. You can use [[Fine-tune|grid or randomized search]] to find the optimal learning rate.
- `n_iter_no_change` enables #early-stopping, automatically stopping training once the validation score has failed to improve by at least `tol` (default `0.0001`) for the specified number of iterations. When enabled, `fit()` automatically holds out a validation set (controlled by `validation_fraction`, 10% by default) to evaluate each new tree. Setting `n_iter_no_change` too low may underfit the data, while setting it too high can lead to overfitting.
- `subsample` specifies the fraction of training instances used to fit each tree. For example, `subsample=0.25` trains every tree on a random 25% of the training instances. This increases bias, reduces variance, speeds up training, and is known as #stochastic-gradient-boosting.
## Histogram-Based Gradient Boosting
#HGB is a #GBRT implementation **optimized for large datasets**. It bins the input features, replacing them with integers. The number of bins is set by `max_bins` (defaults to 255, can't go higher).

Binning greatly reduces the number of thresholds the algorithm must evaluate, allowing HGB to train **hundreds of times faster** than regular GBRT on large datasets. Although binning introduces a small loss of precision, this acts as a **regularizer** that can reduce overfitting, though on some datasets it may instead lead to underfitting.

Scikit-Learn provides `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier`. They differ from the regular GBRT classes in a few ways:

- Early stopping is automatic when there are more than 10,000 instances (force it with `early_stopping=True`/`False`).
- Subsampling is not supported.
- `n_estimators` is renamed `max_iter`.
- Only `max_leaf_nodes`, `min_samples_leaf`, `max_depth`, and `max_features` can be tweaked on the trees.

HGB supports **categorical features** natively — no imputer, scaler, or one-hot encoder needed, you just need to encode them as integers from 0 to a number below `max_bins` — use an `OrdinalEncoder`.

Here's a complete pipeline on the [[Get the Data|California housing dataset]]:

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from my_dataframes import features, labels # defined in ../End-to.../prepare-the-data

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough",
                            force_int_remainder_cols=False),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)

# Note that categorical_features is set to the categorical column indices (or a Boolean array)

hgb_reg.fit(features, labels)
```

> [!tip] When to use HGB
> HGB is a great choice for a **fairly large dataset**, especially with categorical features and missing values, both of which it supports natively: it performs well, needs little preprocessing, and trains fast. It can be slightly less accurate than GBRT due to binning, so it's worth trying both.

