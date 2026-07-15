# PCA
Principal component analysis ( #PCA ) is by far the most popular dimensionality reduction algorithm. It first identifies the **hyperplane that lies closest to the data**, and then **projects the data onto it**.

## How PCA Works

### Preserving the variance
Before you can project the training set onto a lower-dimensional hyperplane, you first need to choose the right hyperplane. 

Consider a 2D dataset and three candidate 1D axes to project onto:

![[7-7.png]]
> Left: A 2D dataset, along with three different axes (i.e., 1D hyperplanes).
> Right: The result of projecting the dataset onto each axis.

Projecting onto the solid line preserves most of the variance, the dotted line preserves very little, and the dashed line falls somewhere in between. Intuitively, it seems reasonable to pick the axis that **preserves the maximum variance**, since it will lose the least information. 

A useful way to visualize this is to think of a shadow. A shadow is simply a projection onto a lower-dimensional surface. If the light is directly overhead, your shadow becomes a small blob that reveals very little about your shape. But if the light comes from the side, your shadow captures much more of your outline. Likewise, PCA looks for the projection that retains as much of the data's structure as possible.

Equivalently, the axis that maximizes the variance is also the one that **minimizes the mean squared distance** between the original data points and their projections.

### Principal Components
PCA finds the axis accounting for the largest amount of variance in the training set. It then finds a second axis, orthogonal to the first, accounting for the largest amount of the *remaining* variance — and a third orthogonal to both, and so on — as many axes as the number of dimensions in the dataset.

The $i$-th axis is called the **$i$-th principal component** of the data. For example, in the figure below, the first two PCs are on the projection plane, and the third PC is the axis orthogonal to that plane.

![[7-2.png]]

After the projection, the first PC corresponds to the $z_1$ axis, and the second PC corresponds to the $z_2$ axis.
![[7-3.png]]

To find the principal components we use a standard matrix factorization technique called **singular value decomposition** ( #SVD ), which decomposes the training set matrix $\mathbf{X}$ into the product $\mathbf{U} \, \mathbf{\Sigma} \, \mathbf{V}^\intercal$, where $\mathbf{V}$ contains the unit vectors defining all the PCs, in order:
$$\mathbf{V} = \begin{pmatrix} \mid & \mid & & \mid \\ \mathbf{c}_1 & \mathbf{c}_2 & \cdots & \mathbf{c}_n \\ \mid & \mid & & \mid \end{pmatrix}$$
$\mathbf{c}_i$ — the unit vector defining the $i$-th principal component.

>[!warning] Direction is not guaranteed
> For each PC, PCA finds a zero-centered unit vector pointing along its direction, but the **sign of that vector isn't stable**: perturb the training set slightly and rerun PCA, and the vector may flip to the opposite direction. A pair of vectors may even rotate or swap if their variances are very close.
>
> So if you use PCA as a preprocessing step before a model, always **retrain the model entirely** whenever you refit the PCA transformer — otherwise the new output won't align with the old and the model will be confused.

### Projecting down to $d$ dimensions
Once you have the PCs, you reduce dimensionality to $d$ dimensions by projecting the data onto the hyperplane defined by the **first $d$ principal components** — this is the projection that preserves the most variance.

To project, multiply the training set $\mathbf{X}$ by $\mathbf{W}_d$, the matrix of the first $d$ columns of $\mathbf{V}$:
$$\mathbf{X}_{d\text{-proj}} = \mathbf{X} \, \mathbf{W}_d$$

## Using Scikit-Learn
Scikit-Learn's `PCA` class uses #SVD under the hood and centers the data:

```python
from sklearn.decomposition import PCA

X = [...] # create a small 3D dataset

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

After fitting, the `components_` attribute holds the transpose of $\mathbf{W}_d$: one row per principal component.

>[!note]
> PCA assumes the dataset is **centered around the origin**. Scikit-Learn's `PCA` classes handle this automatically for you, but if you implement PCA yourself or use another library, don't forget to center the data.

The `explained_variance_ratio_` attribute gives the **proportion of the dataset's variance** lying along each PC:

```python
pca.explained_variance_ratio_ # array([0.82279334, 0.10821224])
```

### Choosing the right number of dimensions
Rather than picking $d$ arbitrarily, it's simpler to choose the number of dimensions that add up to a **sufficiently large portion of the variance** — say 95%. To do this set `n_components` to a float between 0.0 and 1.0 to indicate the ratio of variance to preserve. An exception to this rule, of course, is if you are reducing dimensionality for **data visualization**, in which case you will want to reduce the dimensionality down to 2 or 3.

Let's find the principal components of [[MNIST]] that explain 95% of the variance:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]
```

```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
pca.n_components_ # 154
```

Another option is to **plot the cumulative explained variance** and look for an **elbow** where the explained variance stops growing fast:

```python
pca = PCA()
pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)

[...] # plot cumsum as a function of the number of dimensions.
```

![[7-8.png]]
> Explained variance as a function of the number of dimensions. In this case, you can see that reducing the dimensionality down to about 100 dimensions wouldn’t lose too much explained variance.

>[!tip] Speed and size vs. performance
> Fewer dimensions means a smaller model and faster training/inference. But shrink too much and you lose signal, causing the model to underfit. Choose the balance that fits your use case.

Alternatively, if you are using dimensionality reduction as a **preprocessing step for a supervised learning task**, then you can tune $d$ like any other hyperparameter. The following code creates a two-step pipeline that reduces with PCA and then classifies with a random forest. Next, it uses `RandomizedSearchCV` to find a good combination of hyperparameters for both steps.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
	PCA(random_state=42),
	RandomForestClassifier(random_state=42)
)

param_distrib = {
    "pca__n_components": np.arange(10, 80),
    "randomforestclassifier__n_estimators": np.arange(50, 500)
}

rnd_search = RandomizedSearchCV(
	clf, param_distrib,
	n_iter=10, cv=3, random_state=42
)
rnd_search.fit(X_train[:1000], y_train[:1000])

rnd_search.best_params_
# {'randomforestclassifier__n_estimators': 475, 'pca__n_components': 57}
```

The optimal $d$ is quite low — **784 dimensions down to just 57**. This is tied to the random forest being a powerful model; a linear model like `SGDClassifier` would need more (~75).

>[!note] PCA for Compression
> After reduction the training set takes up **much less space**. Preserving 95% of MNIST's variance leaves **154 features instead of 784** — under 20% of the original size, for only 5% variance lost.
>
> You can also **decompress** back to the original dimensions with the inverse transformation. This won't recover the original data exactly (the dropped 5% of variance is gone), but it'll be close. The mean squared distance between the original and reconstructed data is the **reconstruction error**.
>
> ```python
> X_recovered = pca.inverse_transform(X_reduced)
> ```

---

## PCA Variants
Check this notes on PCA variants:
- [[Randomized PCA]] — a faster stochastic approximation of the first $d$ PCs.
- [[Incremental PCA]] — feeding the data in mini-batches so the whole training set doesn't need to fit in memory.