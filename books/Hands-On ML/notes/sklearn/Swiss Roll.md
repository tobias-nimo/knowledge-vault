# Swiss Roll
The **Swiss roll** is a Scikit-Learn toy dataset used to illustrate **manifold learning** and **nonlinear dimensionality reduction**. It's a 2D plane rolled up in 3D space — like a Swiss roll cake — so the data actually lives on a curved 2D **manifold** embedded in three dimensions.

It's the classic example for showing why simple projections fail: squashing the roll onto a plane just crushes the layers together, whereas manifold learning techniques can **unroll it** back into a flat sheet.

![[7-4.png]]
> The Swiss roll dataset: a 2D manifold rolled up in 3D space.

You generate it with `make_swiss_roll`:

```python
from sklearn.datasets import make_swiss_roll

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
```

- `X_swiss` — the 3D coordinates of each point.
- `t` — a 1D array giving each point's position **along the rolled axis** (i.e. how far along the roll it sits). It isn't a class label, but it makes a natural target for a nonlinear regression task.
- `n_samples` — total number of points generated.
- `noise` — standard deviation of the Gaussian noise added to each point.
- `random_state` — seed for reproducibility.

> Like the other generated toy datasets, you can produce a fresh, independent set just by changing the seed.
