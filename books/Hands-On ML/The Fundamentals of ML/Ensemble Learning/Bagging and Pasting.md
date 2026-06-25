# Bagging and Pasting
Instead of using different algorithms to get diverse predictors (as with [[Voting Classifiers]]), you can use the **same algorithm on different random subsets of the training set**. This can be done with:

- #bagging (short for *bootstrap aggregating*) — samples instances **with replacement**.
- #pasting — samples **without replacement**.

Both let an instance be sampled across multiple predictors, but only bagging lets the same instance be sampled multiple times *for the same predictor*.

Once trained, the ensemble predicts a new instance by aggregating all predictions: 
- **Most frequent** prediction for classification (*statistical mode*)
- **Average** for regression

![[6-4.png]]
> Predictors can be trained — and make predictions — in parallel across CPU cores or servers, allowing bagging and pasting to scale very well.


> [!warning] Use it with high-variance models
> Each individual predictor has a higher bias than if it were trained on the original training set, but aggregation reduces both bias and variance.
> 
> In practice, the ensemble often ends up with a **similar bias but a lower variance** than a single predictor trained on the original training set. Therefore it works best **with high-variance and low-bias models** (e.g., ensembles of decision trees, not ensembles of linear regressors). 

## In Scikit-Learn
Scikit-Learn offers a simple API for both bagging and pasting: 
- `BaggingClassifier` for classification
- `BaggingRegressor` for regression

The following code trains 500 decision trees, each on 100 instances sampled with replacement (bagging) from the [[Moons|moons dataset]]:

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(
	DecisionTreeClassifier(),
	n_estimators=500,
	max_samples=100,
	n_jobs=-1, # CPU cores for training and predictions; -1 uses all available cores
	bootstrap=True, # default: sample with replacement (bagging); set False for pasting
	random_state=42
)

bag_clf.fit(X_train, y_train)

[...] # plot the decision boundary
```

![[6-5.png]]
> A single decision tree (left) versus a bagging ensemble of 500 trees (right). The ensemble has comparable bias but smaller variance — roughly the same number of training errors, but a far **less irregular decision boundary**.

>[!note] Notes
> - A `BaggingClassifier` automatically uses #soft-voting instead of hard voting if the base estimator can estimate class probabilities (has `predict_proba()`).
> - `max_samples` can alternatively be set to a float between `0.0` and `1.0`, in which case the max number of sampled instances is equal to the size of the training set times max_samples.

>[!tip] Bagging or Pasting?
>Bagging adds a bit more diversity to each subset than pasting, so it has slightly higher bias — but the predictors end up less correlated, lowering the ensemble's variance. **Bagging usually gives better models overall**, which is why it's generally preferred.
>
>Prefer bagging when the data is noisy or the model overfits easily (e.g. a deep decision tree). Otherwise prefer pasting: it avoids redundancy during training, making it a bit more efficient.

## Out-of-Bag evaluation
With bagging, each predictor samples $m$ instances with replacement. It can be shown that, as $m$ grows, only about **63%** of instances are sampled on average per predictor — the remaining ~**37%** are the #out-of-bag (OOB) instances (not the same 37% for every predictor).

Because each instance is OOB for several predictors, those predictors can vote on it to give a **fair ensemble prediction** — no separate validation set needed. Set `oob_score=True` to get an automatic OOB evaluation in the `oob_score_` attribute:

```python
bag_clf = BaggingClassifier(
	DecisionTreeClassifier(),
	n_estimators=500,
	oob_score=True,
	n_jobs=-1,
	random_state=42
)

bag_clf.fit(X_train, y_train)
bag_clf.oob_score_ # 0.896 -> OOB estimate for the test accuracy

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred) # 0.92 -> OOB was a little pessimistic
```

The per-instance class probabilities from OOB voting live in `oob_decision_function_`:

```python
bag_clf.oob_decision_function_[:3] # probas for the first 3 instances
```
```
array([[0.32352941, 0.67647059],
       [0.3375    , 0.6625    ],
       [1.        , 0.        ]])
```

## Sampling features
`BaggingClassifier` can **also sample features** using:  
  
- `max_features` — the number (or fraction) of features to sample.  
- `bootstrap_features` — whether features are sampled with replacement.  
  
This gives rise to two methods:

- #random-patches — sample both instances *and* features.  
- #random-subspaces — use all instances (`bootstrap=False`, `max_samples=1.0`) but sample features (`bootstrap_features=True` and/or `max_features < 1.0`).  
  
Each predictor is then trained on a different subset of features, which is especially useful for **high-dimensional datasets** (e.g. images), where it can significantly reduce training time.  
  
Sampling features further increases predictor diversity, typically trading a small increase in bias for lower variance.