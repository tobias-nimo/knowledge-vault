# Ways to Measure Distance
Various distance measures, or **norms**, are possible:
- The **RMSE** uses the **Euclidean norm** (or $ℓ_2$ norm)—this is the notion of distance we are all familiar with: the straight-line distance between two points. $$\|\mathbf{v}\|_2=\sqrt{v_1^2+v_2^2+\cdots+v_n^2}$$
- The **MAE** uses the $ℓ_1$ norm, also known as the **Manhattan norm**—it measures the distance between two points in a city if you can only travel along orthogonal city blocks. $$ \|\mathbf{v}\|_1=|v_1|+|v_2|+\cdots+|v_n|$$
- More generally, the $ℓ_k$ norm of a vector $\mathbf{v}$ with $n$ elements is defined as: $$\|\mathbf{v}\|_k=\left(|v_1|^k+|v_2|^k+\cdots+|v_n|^k\right)^{1/k}$$

Special cases include:
- $\ell_0$ which counts the number of nonzero elements in the vector, and
- $\ell_\infty$ which corresponds to the maximum absolute value in the vector.

The larger the norm index, the more emphasis is placed on large values while smaller values become less important. This is why RMSE is more sensitive to outliers than MAE. However, when outliers are exponentially rare (as in a bell-shaped distribution), RMSE tends to perform well and is generally preferred.