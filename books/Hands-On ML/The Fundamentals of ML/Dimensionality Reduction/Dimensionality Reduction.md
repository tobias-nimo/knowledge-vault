# Dimensionality Reduction
#dimensionality-reduction means **reducing the number of features** used to describe each training instance while preserving as much useful information as possible.

Many machine learning problems involve **thousands or even millions of features**, making models slower to train, more memory-intensive, and often harder to optimize. High-dimensional spaces also make it increasingly difficult to find meaningful patterns, a phenomenon known as the [[Curse of Dimensionality]].

Fortunately, high-dimensional data often contains a lot of **redundancy and noise**. Some features carry little useful information, while others are highly correlated and encode nearly the same information. 

Dimensionality reduction exploits this by removing irrelevant features and combining redundant ones into a smaller set of informative features, reducing noise and mitigating the curse of dimensionality. As a result, models often **train faster** and achieve **better predictive performance**.

>[!warning] Not Always Beneficial
> Reducing dimensionality can also **discard useful information**. Like compressing an image, an aggressive reduction may simplify the data too much and hurt predictive performance. 
> 
> Some models — #neural-networks in particular — handle high-dimensional data fine and learn their own reduction internally, so this extra preprocessing step won't always help.

Another major benefit is **data visualization**. Reducing a dataset to two or three dimensions makes it possible to plot it, helping reveal patterns such as clusters, outliers, or relationships that would otherwise be difficult to see.

There are two main approaches to dimensionality reduction: #projection and #manifold-learning.

## Projection
In most real-world problems, training instances are *not* spread out uniformly across all dimensions. Many features are nearly constant, and others are highly correlated. As a result, all instances lie within (or close to) a much **lower-dimensional subspace** of the high-dimensional space.

For example, the 3D dataset below lies close to a 2D plane:

![[7-2.png]]
> A 3D dataset (small spheres) lying close to a 2D subspace.

If you project every instance perpendicularly onto that subspace (the dashed lines connecting each point to the plane), you get a new 2D dataset. Its axes are new features $z_1$ and $z_2$ — the coordinates of the projections on the plane.

![[7-3.png]]
> The new 2D dataset after projection onto the subspace.

## Manifold Learning
Projection is fast and often works well, but it's not always the best approach. The **subspace may twist and turn**, as in the *Swiss roll* dataset:

![[7-4.png]]
> The Swiss roll: 3D points arranged in a rolled-up sheet.

Simply projecting onto a plane (e.g. by dropping $x_3$) would squash the different layers of the roll together. What you actually want is to **unroll it** into a clean 2D dataset:

![[7-5.png]]
> Squashing by projecting onto a plane (left) versus unrolling the Swiss roll (right).

The *Swiss Roll* is an example of a **2D manifold**: a 2D shape that's been **bent and twisted inside a higher-dimensional space**. More generally, a $d$-dimensional manifold is a part of an $n$-dimensional space (with $d < n$) that locally resembles a $d$-dimensional hyperplane. For the Swiss roll, $d = 2$ and $n = 3$: it locally looks like a 2D plane, but it's rolled up in the third dimension.

Many algorithms ( #LLE, #isomap, #t-SNE, #UMAP ) work by modeling the manifold the instances lie on — this is called **manifold learning**. It relies on the **manifold assumption**: most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This is very often observed empirically.

Think again about [[MNIST]]: handwritten digits are made of connected lines, have white borders, and are roughly centered. If you generated images at random, only a tiny fraction would look like digits. The **degrees of freedom** for producing a digit are far fewer than for producing *any* image, and those constraints squeeze the dataset onto a lower-dimensional manifold.

The manifold assumption usually comes with an implicit companion: that the **task itself will be simpler** when expressed in the lower-dimensional manifold space. For the Swiss roll split into two classes, the decision boundary is a complicated surface in 3D but becomes a straight line once unrolled.

![[7-6.png]]
> The decision boundary may not always be simpler with lower dimensions.
> Top row: a boundary that's complex in 3D becomes a straight line on the unrolled manifold.
> Bottom row: a boundary at $x_1 = 5$ is a simple vertical plane in 3D but breaks into four separate line segments once unrolled.

>[!warning] Assumptions
> That implicit assumption **doesn't always hold**. As the bottom row above shows, a boundary that's trivial in the original space can look *more* complex on the manifold. 
> 
> Reducing dimensionality before training usually speeds things up, but it won't always give a better or simpler solution — it depends on the dataset.

---

Notes on dimensionality reduction techniques:

- [[PCA]]
- [[Random Projection]]
- [[LLE]]
- [[Other Techniques|MDS]]
- [[Other Techniques|Isomap]]
- [[Other Techniques|TSNE]]
- [[Other Techniques|LDA]]
- [[Other Techniques|UMAP]]