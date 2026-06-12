# Handling Non-Symmetrical Distributions
We know ML algorithms like features to be scaled... But if a feature has a **heavy-tailed distribution**, most values may get compressed into a small range after scaling, making it harder for models to learn useful patterns.
## Handling Skewed Distributions
Common transformations include:
- **Square root transformation**: useful for moderately right-skewed positive features.
- **Log transformation**: useful for heavily right-skewed positive features (e.g., power-law distributions).
- **Power transformations**: raising values to a power between 0 and 1 can reduce skewness.

These transformations often make the distribution more symmetric and **closer to a Gaussian (bell-shaped) distribution**.
![[2-17.png]]

> Another approach is **bucketization** (or binning), which divides a feature into discrete intervals (buckets). For example, replacing each value with its percentile rank creates a feature with an approximately uniform distribution. If bucket indices are treated as numerical values, scaling is usually unnecessary. Dividing by the number of buckets can normalize them to the 0–1 range.
## Handling Multimodal Distributions
Features with **multiple peaks** (multimodal distributions) can benefit from **bucketization** as well. In this case, bucket IDs are often treated as categorical variables and encoded using techniques such as #one-hot-encoding. 

This allows models to learn different relationships for different value ranges instead of assuming a single continuous trend across the entire feature.

Another way to handle this distributions is to create new features that measure similarity to each of the modes (at least the main ones). The similarity measure is typically computed using a **Radial Basis Function**, like the Gaussian RBF:
$$exp⁡(−γ(x−c)^2)$$
-  $c$ is the center (e.g., a mode of the distribution).
- $γ$ (gamma) controls how quickly similarity decreases with distance.

> An RBF is any function that depends only on the distance between the input value and a fixed point.

Using Scikit-Learn:
```python
from sklearn.metrics.pairwise import rbf_kernel

# Create a new Gaussian RBF feature measuring the similarity between a multimodal variable and one of it's modes (35)
var1_similarity_35 = rbf_kernel(feature[["var1"]], [[35]], gamma=0.1)
```
The resulting feature has values close to 1 near the chosen center and approaches 0 as values move away from it.
![[2-18.png]]