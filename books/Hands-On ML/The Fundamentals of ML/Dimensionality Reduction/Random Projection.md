# Random Projection
#random-projection projects the data to a lower-dimensional space using a **random linear projection**. It sounds crazy, but such a projection is very likely to **preserve distances fairly well**: two similar instances stay similar after the projection, and two very different instances stay very different.

> [!note] Johnson-Lindenstrauss Lemma
> This distance-preserving property was proven mathematically by Johnson and Lindenstrauss in a famous lemma.

The more dimensions you drop, the more information is lost and the more distances get distorted. The Johnson–Lindenstrauss lemma gives the minimum number of dimensions $d$ you must keep to ensure — with high probability — that squared distances won't change by more than a tolerance $\varepsilon$:
$$d \geq \frac{4 \log(m)}{\frac{1}{2}\varepsilon^2 - \frac{1}{3}\varepsilon^3}$$
Where
- $m$ is the number of instances.
- $\varepsilon$ is the maximum tolerated change in squared distance between any two instances.

Notice the equation **does not depend on the number of features** $n$, only on $m$ and $\varepsilon$. For example, with $m = 5{,}000$ instances of $n = 20{,}000$ features each and $\varepsilon = 10\%$, you should project down to $d = 7{,}300$ dimensions — a significant reduction. 

This is implemented by Scikit-Learn's `johnson_lindenstrauss_min_dim()`:

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim

m, ε = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=ε) # 7300
```

## Doing it by hand
First, we generate a random matrix $P$ of shape `[d, n]`, where each item is sampled from a Gaussian with mean 0 and variance $1/d$. Projecting the data is then just a matrix multiplication:

```python
import numpy as np

n = 20_000
rng = np.random.default_rng(seed=42)
P = rng.standard_normal((d, n)) / np.sqrt(d)  # std dev = sqrt(variance)

X = rng.standard_normal((m, n))  # generate a fake dataset
X_reduced = X @ P.T
```

That's all there is to it. Training is **almost instantaneous**: the only thing the algorithm needs to build the random matrix is the dataset's *shape* — the data itself is never used. 

This makes random projection particularly well suited for very high-dimensional data (text, genomics with millions of features) or very sparse data, where even #randomized-pca may be too slow or memory-hungry. At inference time it is just as fast as #PCA (one matrix multiplication).

> [!tip] Random Projection vs. PCA
> Random projection is a **simple, fast, memory-efficient, and surprisingly powerful** dimensionality reduction technique for high-dimensional datasets. 
> 
> The **trade-off** is that it typically preserves slightly less information than PCA, sacrificing a bit of performance in exchange for much faster training.

## In Scikit-Learn
Scikit-Learn's **`GaussianRandomProjection`** implements exactly the algorithm described above. During `fit()`, it computes the target dimensionality, generates the random projection matrix, and stores it in `components_`. Calling `transform()` simply applies this matrix to the data.

```python
from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)  # same result as above

# gaussian_rnd_proj.components_ is equal to P above
```

You can set `eps` to control the tolerated distortion $\varepsilon$ (default `0.1`), or specify the target dimensionality directly using `n_components` — you'll probably want to fine-tune these with #cross-validation.

Scikit-Learn also provides **`SparseRandomProjection`**, which chooses the target dimensionality the same way but uses a **sparse** random projection matrix instead of a dense Gaussian one. It enjoys the same distance-preserving guarantees, while requiring much less memory and computation. Moreover, if the input is sparse, the output remains sparse as well (unless `dense_output=True`).

```python
from sklearn.random_projection import SparseRandomProjection

sparse_rnd_proj = SparseRandomProjection(eps=ε, random_state=42)
X_reduced = sparse_rnd_proj.fit_transform(X)
```

> [!note] Density
> The ratio $r$ of nonzero items in the sparse matrix is its density. 
> 
> By default $r = \frac{1}{\sqrt{n}}$ — with 20,000 features, only about 1 in ~141 cells is nonzero. Each cell is nonzero with probability $r$, and each nonzero value is either $-v$ or $+v$ (equally likely), where $v = \frac{1}{\sqrt{dr}}$. 
> 
> You can override this with the `density` hyperparameter.

> [!tip]  Gaussian vs. Sparse
> In practice, `SparseRandomProjection` is usually the preferred implementation for large or sparse datasets.

## Inverse transform
To approximately reconstruct the original space, compute the **pseudoinverse** of the components matrix, then multiply the reduced data by its transpose:

```python
P = gaussian_rnd_proj.components_
components_pinv = np.linalg.pinv(P)
X_recovered = X_reduced @ components_pinv.T
```

>[!warning]
> Computing the pseudoinverse can be very slow for a large projection matrix.