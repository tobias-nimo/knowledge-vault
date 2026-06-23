# Moons Dataset
The **moons dataset** is a #sklearn toy dataset for **binary classification** in which the data points are shaped as **two interleaving crescent moons**. Because the two classes are not linearly separable, it's a handy example for testing models that can carve out nonlinear decision boundaries.

You generate it on the fly with `make_moons`:

```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)
```

- `n_samples` — total number of points generated.
- `noise` — standard deviation of the Gaussian noise added to each point. Higher values blur the two crescents into each other.
- `random_state` — seed for reproducibility.

> Since it's generated from a function rather than loaded, you can produce a fresh, independent test set just by changing the seed.