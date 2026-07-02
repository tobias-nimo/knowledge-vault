# Randomized PCA
#randomized-pca is a stochastic algorithm that quickly finds an **approximation** of the first $d$ principal components, instead of computing the exact ones via full [[PCA|SVD]].

You enable it by setting the `svd_solver` hyperparameter to `"randomized"`:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]
```

```python
from sklearn.decomposition import PCA

rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

It's **dramatically faster** than the full #SVD approach.

>[!note] The "auto" solver
> By default `svd_solver="auto"`, and Scikit-Learn picks a solver for you:
> - If the data has few features ($n < 1{,}000$) and at least 10× more samples ($m > 10n$), it uses the very fast `"covariance_eigh"` solver.
> - Otherwise, if $\max(m, n) > 500$ and `n_components` is an integer smaller than 80% of $\min(m, n)$, it uses `"randomized"`.
> - In other cases it falls back to full SVD.
>
> To force full SVD — trading compute time for a slightly more precise result — set `svd_solver="full"`.

>[!warning] When PCA is too slow
> For datasets with tens of thousands of features or more (e.g. images), even randomized PCA training can become far too slow — consider [[Random Projection]] instead.
