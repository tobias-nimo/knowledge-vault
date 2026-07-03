# LLE
**Locally linear embedding** ( #LLE ) is a *nonlinear dimensionality reduction* technique. It is a #manifold-learning technique that does not rely on projections, unlike PCA and random projection.

In a nutshell, LLE first measures how each training instance **linearly relates to its nearest neighbors**, then looks for a low-dimensional representation of the training set where these **local relationships are best preserved**. This makes it particularly good at unrolling twisted manifolds, especially when there isn't too much noise.

>[!warning] Scaling
> LLE doesn't scale well, so it's mostly suited to small or medium-sized datasets.

## In Scikit-Learn
The following code builds a [[Swiss Roll]] and uses `LocallyLinearEmbedding` to unroll it:

```python
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
# we don't use t here

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)
```

![[7-10.png]]
> Unrolled Swiss roll using LLE. 

The roll is completely unrolled, and distances between instances are **locally well preserved**. On a larger scale, though, distances are not preserved — the result should be a rectangle, not this stretched and twisted band. Still, LLE modeled the manifold quite well.

> [!tip] 
> Overall, LLE is quite different from the projection techniques and significantly more complex, but it can build **much better low-dimensional representations**, especially when the data is nonlinear.

## How LLE works
LLE runs in two steps.

**Step 1 — model local relationships.** For each instance $\mathbf{x}^{(i)}$, find its $k$-nearest neighbors (here $k = 10$), then reconstruct $\mathbf{x}^{(i)}$ as a **linear combination** of those neighbors. Concretely, find the weights $w_{i,j}$ that minimize the squared distance between $\mathbf{x}^{(i)}$ and $\sum_j w_{i,j}\mathbf{x}^{(j)}$, with $w_{i,j} = 0$ whenever $\mathbf{x}^{(j)}$ is not a neighbor of $\mathbf{x}^{(i)}$:
$$\hat{\mathbf{W}} = \underset{\mathbf{W}}{\text{argmin}} \sum_{i=1}^{m} \left\lVert \mathbf{x}^{(i)} - \sum_{j=1}^{m} w_{i,j}\mathbf{x}^{(j)} \right\rVert^2$$
subject to
$$\begin{cases} w_{i,j} = 0 & \text{if } \mathbf{x}^{(j)} \text{ is not one of the } k \text{ nearest neighbors of } \mathbf{x}^{(i)} \\ \sum_{j=1}^{m} w_{i,j} = 1 & \text{for } i = 1, 2, \dots, m \end{cases}$$
Where
- $\mathbf{W}$ — the weight matrix holding all weights $w_{i,j}$.
- The second constraint simply **normalizes** the weights for each instance.

After this step, $\mathbf{W}$ encodes the local linear relationships between the training instances.

**Step 2 — reduce dimensionality.** Map the instances into a $d$-dimensional space ($d < n$) while preserving those relationships as much as possible. If $\mathbf{z}^{(i)}$ is the image of $\mathbf{x}^{(i)}$, keep the weights fixed and find the image positions that minimize the squared distance between $\mathbf{z}^{(i)}$ and $\sum_j w_{i,j}\mathbf{z}^{(j)}$:
$$\hat{\mathbf{Z}} = \underset{\mathbf{Z}}{\text{argmin}} \sum_{i=1}^{m} \left\lVert \mathbf{z}^{(i)} - \sum_{j=1}^{m} w_{i,j}\mathbf{z}^{(j)} \right\rVert^2$$
Where $\mathbf{Z}$ is the matrix holding all images $\mathbf{z}^{(i)}$.

This is the mirror image of step 1: instead of fixing the instances and solving for the weights, we fix the weights and solve for the low-dimensional positions.
