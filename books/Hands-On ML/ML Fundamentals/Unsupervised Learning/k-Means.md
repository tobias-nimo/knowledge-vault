# k-Means
#k-means is a simple algorithm capable of **clustering blob-like datasets** very quickly and efficiently, often in just a few iterations: it finds each blob's center and assigns each instance to the closest one.

Consider this unlabeled dataset with five clearly visible blobs:

![[8-2.png]]

Let's train a `KMeans` clusterer on it:

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs([...]) # y contains the cluster IDs, but we won't use them

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
```

Note that **you must specify the number of clusters** $k$ the algorithm has to find. Here it's obviously 5, but in general it isn't that easy (see [[#Finding the Optimal Number of Clusters]]).

The predicted labels of the training instances are kept in `labels_`, and the centroids the algorithm found in `cluster_centers_`:

```python
y_pred # array([4, 0, 1, ..., 2, 1, 0], dtype=int32)
y_pred is kmeans.labels_ # True

kmeans.cluster_centers_
# array([[-2.80389616,  1.80117999],
#        [ 0.20876306,  2.25551336],
#        ...
#        [-2.80037642,  1.30082566]])
```

New instances simply get assigned to the cluster whose centroid is closest:

```python
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new) # array([1, 1, 2, 2], dtype=int32)
```

Assigning each instance to a single cluster with `predict()` is called #hard-clustering ; giving each instance a score per cluster (a distance, or a similarity score) is called #soft-clustering. The `transform()` method measures the distance from each instance to every centroid:

```python
kmeans.transform(X_new).round(2)
# array([[2.81, 0.33, 2.9 , 1.49, 2.89],
#        [5.81, 2.8 , 5.85, 4.48, 5.84],
#        [1.21, 3.29, 0.29, 1.69, 1.71],
#        [0.73, 3.22, 0.36, 1.55, 1.22]])
```

>[!tip] Nonlinear dimensionality reduction
> If you have a high-dimensional dataset and you transform it this way, you end up with a k-dimensional dataset: this transformation can be a very efficient nonlinear #dimensionality-reduction technique. 

> [!tip] Create new features
> Alternatively, you can use the computed distances as **extra features** to train another model, as we did [[Custom Transformers|here]]. #feature-engineering

Plotting the decision boundaries gives the following figure:

![[8-3.png]]
> k-means decision boundaries; each centroid is marked with an ×. Most instances get assigned to the right cluster, but a few near the boundaries are probably mislabeled.

> [!warning]
> k-means **only cares about the distance to the centroid**, so it doesn't behave well when blobs have very different diameters.

## The Algorithm
1. **Choose $k$.** Decide in advance how many clusters you want. This is a hyperparameter you supply; the algorithm doesn't figure it out on its own.
2. **Initialize centroids.** Pick $k$ data points at random from your dataset and use their positions as the starting centroids.
3. **Assign each point to a cluster.** For every data point, compute its distance to each of the $k$ centroids (usually [[Ways to Measure Distance|Euclidean distance]]):
	$$\left\| \mathbf{x}^{(i)} - \mathbf{c}^{(j)} \right\|^2$$
	Then assign each point to its nearest centroid:
	$$z^{(i)} = \arg\min_j \left\| \mathbf{x}^{(i)} - \mathbf{c}^{(j)} \right\|^2$$
	where $\mathbf{x}^{(i)}$ is the $i$-th instance, $\mathbf{c}^{(j)}$ is the $j$-th centroid, and $z^{(i)}$ is the cluster index assigned to the $i$-th instance.
4. **Update each centroid.** For each cluster, collect all instances assigned to it and move the centroid to their mean position — i.e., average each coordinate independently across those instances. A centroid is simply the geometric center of its current members.
5. **Check for convergence.** Compare the new centroid positions to the old ones. If any centroid moved (beyond some tiny tolerance), go back to step 3 and repeat. If none moved, you're done. 

The final centroid positions represent the center of each discovered group, and each instance has a final cluster label. 

The algorithm is **guaranteed to converge** in a finite (usually small) number of steps: at each iteration, both step 3 and step 4 can only decrease or maintain the mean squared distance between instances and their closest centroids.

![[8-4.png]]
> The k-means algorithm: centroids initialized randomly (top left), instances labeled (top right), centroids updated (center left), and so on — close to optimal in just three iterations.

>[!tip] Computational complexity
> k-means is generally **one of the fastest clustering algorithms**.

## Centroid Initialization
Although convergence is guaranteed, it is **not necessarily guaranteed to the best solution**: depending on the random initialization, it may converge to a *local optimum*.

![[8-5.png]]
> Two suboptimal solutions caused by unlucky centroid initializations.

There are a few ways to mitigate the risk of a bad initialization:

- If you know roughly where the centroids should be (e.g. from an earlier clustering), pass them via the `init` hyperparameter:

	```python
	good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
	kmeans = KMeans(n_clusters=5, init=good_init, random_state=42)
	kmeans.fit(X)
	```

- Another solution is to run the algorithm multiple times with **different random initializations** and keep the best solution. The number of random initializations is controlled by the `n_init` hyperparameter: by default it is equal to 10 when using `init="random"`.

To know which solution is *best*, Scikit-Learn uses the model's **inertia**: the sum of squared distances between each instance and its closest centroid:
$$\text{inertia} = \sum_{i} \left\| \mathbf{x}^{(i)} - \mathbf{c}^{(i)} \right\|^2$$
- $\mathbf{x}^{(i)}$ — the $i$-th instance.
- $\mathbf{c}^{(i)}$ — the closest centroid to the $i$-th instance.

The `KMeans` class runs `n_init` initializations and keeps the model with the **lowest inertia**, accessible via `inertia_`:

```python
kmeans.inertia_ # 211.59853725816828
kmeans.score(X) # -211.59853725816828

# remember that score() returns the negative inertia, respecting Scikit-Learn's "greater is better" rule for predictors.
```

>[!note] k-means++
> An important improvement to the k-means algorithm was proposed in a 2006 paper, [k-means++](https://scholar.google.com/scholar?q=k-means%2B%2B%3A+The+advantages+of+careful+seeding+author%3Aarthur). It introduces a **smarter initialization step that tends to select centroids distant from one another**, making the algorithm much less likely to converge to a suboptimal solution.
> 
> When you set `init="k-means++"` (which is the default), the `KMeans` class actually uses this variant. When using this algorithm, `n_init` defaults to 1.

## Mini-Batch k-Means
[Mini-batch k-means](https://scholar.google.com/scholar?q=Web-Scale+K-Means+Clustering+author%3Asculley) updates centroids using **small random batches** instead of the full dataset at each iteration. This makes it **much faster and more memory-efficient** than standard k-means, making it well suited for **very large datasets** that may not fit in memory. In exchange, it typically sacrifices only a small amount of clustering quality.

```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
```

>[!tip] 
>If the dataset does not fit in memory, the simplest option is to use `np.memmap` as we did [[Incremental PCA|here]].

> [!note] Elkan's k-Means
> [Elkan's k-means](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf) (`algorithm="elkan"`) is another variant that speeds up training by avoiding unnecessary distance calculations. It can be significantly faster on large datasets with many clusters, but depending on the case it may even be slower.

## Finding the Optimal Number of Clusters
So far we've set $k = 5$ because it was obvious from looking at the data, but in general it won't be so easy to know the correct number of clusters — and the result can be quite bad if you set it to the wrong value:

![[8-6.png]]
> Too small a k merges separate clusters (left); too large a k chops clusters into pieces (right).

You can't just pick the $k$ with the lowest inertia: **inertia keeps decreasing as $k$ increases** (more clusters means each instance is closer to its centroid). Instead, use the **silhouette score**: the mean *silhouette coefficient* over all instances.
$$\text{silhouette coefficient} = \frac{b - a}{\max(a, b)}$$
Where:
- $a$ — mean distance to the other instances in the same cluster (mean intra-cluster distance).
- $b$ — mean distance to the instances of the next closest cluster.

The silhouette coefficient varies between **–1 and +1**: close to **+1** means the instance is well inside its own cluster and far from others; close to **0** means it's near a cluster boundary; close to **–1** means it may have been assigned to the **wrong cluster**.

```python
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_) # 0.655517642572828

[...] # plot silhouette scores for different numbers of clusters
```

![[8-8.png]]
> k = 4 is very good, but k = 5 is quite good too, and much better than 6 or 7.

Even more informative is the **silhouette diagram**: every instance's silhouette coefficient, sorted by cluster and by coefficient value. Each cluster is a *knife shape* — its **height** is the number of instances, its **width** the sorted coefficients (wider is better).

![[8-9.png]]
> Silhouette diagrams for various k. When most of a cluster's instances stop short of the mean silhouette score (dashed line), the cluster is bad — its instances are much too close to other clusters.

Here $k = 3$ and $k = 6$ give bad clusters, while $k = 4$ and $k = 5$ look good. Even though $k = 4$ has a slightly higher overall score, $k = 5$ yields clusters of similar sizes — so it seems like the better choice.

## Limits of k-Means
Despite being **fast and scalable**, k-means is not perfect:

- It must be **run several times** to avoid suboptimal solutions.
- You need to **specify the number of clusters**.
- It doesn't behave well when clusters have **varying sizes, different densities, or non-spherical shapes**.

![[8-10.png]]
> k-means fails to cluster three ellipsoidal blobs properly: the left solution chops off 25% of the middle cluster; the right one is terrible despite having a lower inertia.

Depending on the data, different algorithms may perform better — on elliptical clusters, [[Gaussian Mixtures|Gaussian mixture models]] work great.

>[!warning] Scale your features!
> Always scale the input features before running k-means, or the clusters may be very stretched and k-means will perform poorly. Scaling doesn't guarantee nice spherical clusters, but it generally helps.
