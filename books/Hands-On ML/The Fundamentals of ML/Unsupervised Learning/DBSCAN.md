# DBSCAN
#DBSCAN (*density-based spatial clustering of applications with noise*) defines clusters as **continuous regions of high density**. By estimating the local density around each instance, it can identify **clusters of arbitrary shapes** and naturally detect **outliers**.

## The Algorithm
- For each instance, count how many instances are located within a small distance $\varepsilon$ (epsilon) from it. This region is called the instance's **ε-neighborhood**.
- If an instance has at least `min_samples` instances in its ε-neighborhood (including itself), it is a **core instance** — an instance located in a dense region.
- All instances in the neighborhood of a core instance belong to the same cluster. That neighborhood may contain other core instances, so a long chain of neighboring core instances forms a single **cluster**.
- Any instance that is not a core instance and has none in its neighborhood is considered an **anomaly**.

This works well as long as all the clusters are **separated by low-density regions**.

## In Scikit-Learn
Let's test the `DBSCAN` class on the [[Moons|moons dataset]]:

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

The labels of all instances are in `labels_`. **A label of –1 means the instance is considered an anomaly**:

```python
dbscan.labels_ # array([ 0, 2, -1, -1, 1, 0, 0, 0, 2, 5, [...], 3, 3, 4, 2, 6, 3])
```

The indices of the core instances are in `core_sample_indices_`, and the core instances themselves in `components_`:

```python
dbscan.core_sample_indices_ # array([ 0, 4, 5, 6, 7, 8, 10, 11, [...], 995, 997, 998, 999])

dbscan.components_
# array([[-0.02137124,  0.40618608],
#        [-0.84192557,  0.53058695],
#        [...],
#        [ 0.79419406,  0.60777171]])
```

With `eps=0.05` the result is disappointing: lots of anomalies and seven different clusters. Widening each instance's neighborhood by increasing `eps` to 0.2 gives a perfect clustering:

![[8-13.png]]
> DBSCAN clustering using two different neighborhood radiuses: eps = 0.05 (left) finds 7 clusters and many anomalies; eps = 0.2 (right) looks perfect.

## Predicting New Instances

>[!note] No predict method
> Surprisingly, `DBSCAN` has a `fit_predict()` method but no `predict()`: it cannot tell which cluster a *new* instance belongs to. The authors decided to let you choose the classification algorithm that best fits your task.

It's easy to implement yourself — for example, train a `KNeighborsClassifier` on the core instances:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new) # array([1, 0, 1, 0])
knn.predict_proba(X_new)
# array([[0.18, 0.82],
#        [1.  , 0.  ],
#        [0.12, 0.88],
#        [1.  , 0.  ]])
```

We trained the classifier only on the **core instances**, but we could also have used all the instances, or all but the anomalies — the choice depends on the final task.

![[8-14.png]]
> Decision boundary between the two clusters (the crosses are the four instances in X_new).

Since there is no anomaly in the classifier's training set, it **always picks a cluster**, even for instances far away from both. To fix this, introduce a **maximum distance** using the `kneighbors()` method, which returns the distances and indices of the k-nearest training instances:

```python
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1 # farther than 0.2 from any core instance -> anomaly
y_pred.ravel() # array([-1, 0, 1, -1])
```

## Strengths and Limits
DBSCAN is a **simple yet powerful** algorithm:

- It can identify **any number of clusters of any shape**.
- It is **robust to outliers**.
- It has just **two hyperparameters** (`eps` and `min_samples`).

However:

- If the **density varies significantly** across clusters, or there's no sufficiently low-density region around some of them, it can struggle to capture all the clusters properly.
- Its computational complexity is roughly $O(m^2 n)$, so it **does not scale well to large datasets**.

>[!tip] HDBSCAN
> Try *hierarchical DBSCAN* (`sklearn.cluster.HDBSCAN`): it is often better than DBSCAN at finding **clusters of varying densities**.
