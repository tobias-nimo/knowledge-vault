# Gaussian Mixtures
A Gaussian mixture model ( #GMM ) is a **probabilistic model** that assumes the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. All the instances generated from a single Gaussian distribution form a cluster that typically looks like an **ellipsoid** — each with its own shape, size, density, and orientation. GMMs can be used for density estimation, clustering, and anomaly detection.

The dataset $\mathbf{X}$ is assumed to have been generated through this **probabilistic process**:
- For each instance, a cluster is picked randomly among the $k$ clusters. The probability of choosing the $j$-th cluster is the **cluster's weight** $\phi^{(j)}$. The index of the cluster chosen for the $i$-th instance is denoted $z^{(i)}$.
- If the $i$-th instance was assigned to the $j$-th cluster (i.e., $z^{(i)} = j$), then the location $\mathbf{x}^{(i)}$ of this instance is **sampled from the Gaussian distribution** with mean $\boldsymbol{\mu}^{(j)}$ and covariance matrix $\boldsymbol{\Sigma}^{(j)}$:
$$\mathbf{x}^{(i)} \sim \mathcal{N}\left(\boldsymbol{\mu}^{(j)}, \boldsymbol{\Sigma}^{(j)}\right)$$
When you observe an instance you know it came from one of the Gaussians, but you don't know *which one*, nor the distributions' parameters.

## Expectation-Maximization Algorithm
GMM relies on the expectation-maximization ( #EM ) algorithm:

1. **Choose k.** Decide in advance how many Gaussian components you want.
2. **Initialize cluster parameters randomly.** Each cluster is characterized by three parameters:
	- $\boldsymbol{\mu}^{(j)}$: the mean (center) of the cluster
	- $\boldsymbol{\Sigma}^{(j)}$: the covariance matrix (encoding size, shape, and orientation)
	- $\phi^{(j)}$: the weight (relative proportion of instances belonging to it)
3. **Expectation step**. For every instance, estimate the probability that it belongs to each cluster given the current parameters. These probabilities are called the **responsibilities** of each cluster for each instance:
	$$r^{(i,j)} = P(\text{cluster } j \mid \mathbf{x}^{(i)})$$
4. **Maximization step**. Update each cluster's parameters using all instances in the dataset, but weighting each instance by its responsibility $r^{(i,j)}$.
5. **Check for convergence.** Compare the new parameters to the old ones. If they moved beyond some tiny tolerance, go back to step 3 and repeat. If not, you're done.

>[!note]
> The exact update formulas for these steps are derived via [[Likelihood|maximum likelihood estimation]]: the maximization step sets each cluster's parameters to the values that maximize the (responsibility-weighted) likelihood of the data.

The algorithm is **guaranteed to converge** in a finite (usually small) number of steps: at each iteration, both the expectation and maximization steps are guaranteed to increase (or maintain) the likelihood of the data under the model — never decrease it. Since the likelihood is bounded above, the algorithm must converge.

## In Scikit-Learn
Let's generate the same dataset as earlier with three ellipsoids (the one [[k-Means#Limits of k-Means|k-means]] had trouble with):

```python
import numpy as np 
from sklearn.datasets import make_blobs

X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]
```

Given the dataset $\mathbf{X}$, you typically start by estimating the weights $\phi$ and all the distribution parameters $\boldsymbol{\mu}^{(1)}$ to $\boldsymbol{\mu}^{(k)}$ and $\boldsymbol{\Sigma}^{(1)}$ to $\boldsymbol{\Sigma}^{(k)}$:

```python
from sklearn.mixture import GaussianMixture

# note that you must know in advance the number k of Gaussian distributions
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)
```

The estimated parameters:

```python
gm.weights_ # array([0.40005972, 0.20961444, 0.39032584])

gm.means_
# array([[-1.40764129,  1.42712848],
#        [ 3.39947665,  1.05931088],
#        [ 0.05145113,  0.07534576]])

gm.covariances_
# array([[[ 0.63478217,  0.72970097],
#         [ 0.72970097,  1.16094925]],
#        ...
#        [[ 0.68825143,  0.79617956],
#         [ 0.79617956,  1.21242183]]])
```

The true cluster weights were 0.4, 0.4, and 0.2, and the algorithm found them almost exactly (in a different order), along with means and covariances quite close to the true ones.

>[!warning] Convergence
> Just like the [[k-Means#Centroid Initialization|k-means]] algorithm, EM can converge to suboptimal solutions, so it needs to be **run several times**, keeping only the best solution. That's why we set `n_init=10` — by default it is 1.
> 
> You can check whether the algorithm converged and how many iterations it took:
> ```python
> gm.converged_ # True
> gm.n_iter_ # 4
> ```

With the parameters estimated, the model can assign each instance to its most likely cluster using `predict()` — #hard-clustering — or return the probability of belonging to each cluster using `predict_proba()` — #soft-clustering, which outputs the full probability vector $r^{(i,j)}$ for every instance.

```python
gm.predict(X) # array([2, 2, 0, ..., 1, 1, 1])

gm.predict_proba(X).round(3)
# array([[0.   , 0.023, 0.977],
#        [0.001, 0.016, 0.983],
#        [1.   , 0.   , 0.   ],
#        ...,
#        [0.   , 1.   , 0.   ]])
```

A GMM is a #generative-model, meaning you can sample new instances from it:

```python
X_new, y_new = gm.sample(6)
y_new # array([0, 0, 1, 1, 1, 2]) -> ordered by cluster index
```

It is also possible to **estimate the density** of the model at any given location. This is achieved using the `score_samples()` method: for each instance it is given, this method estimates the log of the probability density function ( #PDF ) at that location. The greater the score, the higher the density:

```python
gm.score_samples(X).round(2) 
# array([-2.61, -3.57, -3.33, ..., -3.51, -4.4 , -3.81])
```

>[!note] Densities, not probabilities
>If you compute the exponential of these scores, you get the value of the PDF at the location of the given instances. These are not probabilities, but probability densities: they can take on any positive value. 
>
>To estimate the probability that an instance will fall within a particular region, you would have to integrate the PDF over that region (if you do so over the entire space of possible instance locations, the result will be 1).

![[8-15.png]]
> Cluster means, decision boundaries (dashed lines), and density contours of a trained Gaussian mixture model.

### Constraining the Covariance Matrices

> [!warning] 
> When there are many dimensions, many clusters, or few instances, EM can struggle to converge to the optimal solution. 

You can reduce the difficulty of the task by **limiting the range of shapes and orientations that the clusters can have**. This can be achieved by imposing constraints on the covariance matrices, via the `covariance_type` hyperparameter:

- `"spherical"` — all clusters must be spherical, but can have different diameters (variances).
- `"diag"` — any ellipsoidal shape and size, but the axes must be parallel to the coordinate axes (diagonal covariance matrices).
- `"tied"` — all clusters share the same shape, size, and orientation (one common covariance matrix).
- `"full"` (default) — each cluster gets its own unconstrained covariance matrix, which means that they can take on any shape, size, and orientation.

![[8-16.png]]
> Gaussian mixtures for tied clusters (left) and spherical clusters (right).

>[!warning] Computational complexity
> `"tied"` and `"full"` won't scale to large numbers of features.

## Anomaly Detection
Using a GMM for #anomaly-detection is simple: **any instance located in a low-density region can be considered an anomaly**. You must define the density threshold. For example, if a manufacturer knows that about 2% of products are defective, set the threshold so that 2% of instances fall below it:

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 2)
anomalies = X[densities < density_threshold]
```

Too many false positives → lower the threshold; too many false negatives → increase it. This is the usual [[Performance Measures#Precision/Recall Curve|precision/recall trade-off]].

![[8-17.png]]
> Anomaly detection using a Gaussian mixture model (anomalies shown as stars).

> [!tip] Improve robustness by removing outliers
> Gaussian Mixture Models try to explain every training instance, including outliers. If there are too many, they can distort the estimated distributions, causing some anomalies to appear normal.
> 
> A common approach is to fit the model once, remove the most extreme outliers, and fit it again on the cleaned dataset. In practice, outlier detection is often used as a #data-cleaning step before training a final model.

## Selecting the Number of Clusters
Here we can't use the [[k-Means#Finding the Optimal Number of Clusters|inertia]] or the [[k-Means#Finding the Optimal Number of Clusters|silhouette score]], as those metrics are not reliable when clusters are non-spherical or have different sizes. Instead, find the model that minimizes a theoretical *information criterion*:
$$\text{BIC} = \log(m)\,p - 2\log(\hat{L})$$
$$\text{AIC} = 2p - 2\log(\hat{L})$$
- $m$ — the number of instances.
- $p$ — the number of parameters learned by the model.
- $\hat{L}$ — the maximized value of the [[Likelihood|likelihood function]] of the model.

Both the #BIC (*Bayesian information criterion*) and the #AIC (*Akaike information criterion*) **penalize models with more parameters to learn** (e.g. more clusters) and **reward models that fit the data well**. They often select the same model; when they differ, the BIC's pick tends to be simpler (fewer parameters) but to fit the data slightly less well, especially on larger datasets.

```python
gm.bic(X) # 8189.733705221636
gm.aic(X) # 8102.508425106598
```

![[8-19.png]]
> AIC and BIC for different numbers of clusters k: both are lowest when k = 3, so it is most likely the best choice.

## Bayesian GMMs
Rather than manually searching for the optimal number of clusters, the `BayesianGaussianMixture` class can give weights equal or close to zero to unnecessary clusters. Set `n_components` to a value you have good reason to believe is greater than the optimal number of clusters, and the algorithm eliminates the unnecessary ones automatically:

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=500, random_state=42)
bgm.fit(X)

bgm.weights_.round(2) 
# array([0.4 , 0.21, 0.39, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])
```

Only three clusters get non-zero weights, and they are almost identical to the ones found before.

## Limits of GMMs

> [!warning] 
> GMMs work great on **ellipsoidal clusters**, but struggle with clusters of very different shapes.

For example, let's see what happens if we use a Bayesian Gaussian mixture model to cluster the [[Moons|moons dataset]]:

![[8-20.png]]
> Fitting a Gaussian mixture to non-ellipsoidal clusters.

Oops! The algorithm desperately searches for ellipsoids and finds eight clusters instead of two. The density estimation is not too bad, but it fails to identify the two moons; use an algorithm like [[DBSCAN]] instead.