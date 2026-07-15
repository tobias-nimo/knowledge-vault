# Other Dimensionality Reduction Techniques
Beyond [[PCA]], [[Random Projection|random projection]], and [[LLE]], Scikit-Learn offers several other popular dimensionality reduction techniques. Here are the ones worth knowing:

- #MDS — Multidimensional scaling (`sklearn.manifold.MDS`) reduces dimensionality while trying to **preserve the distances between instances**. Random Projection does the same for high-dimensional data, but MDS is the better choice on **low-dimensional** data, where random projection doesn't work well.

- #Isomap — Isomap (`sklearn.manifold.Isomap`) builds a graph by connecting each instance to its nearest neighbors, then reduces dimensionality while preserving the **geodesic distances** between instances — the geodesic distance between two nodes being the number of nodes on the shortest path between them.

	It works best when the data lies on a fairly **smooth, low-dimensional manifold with a single global structure** (e.g. the [[Swiss Roll]]).

- #t-SNE — **t-distributed stochastic neighbor embedding** (`sklearn.manifold.TSNE`) reduces dimensionality while keeping **similar instances close and dissimilar instances apart**. It's mostly used for **visualization**, in particular to reveal clusters of instances in high-dimensional space — for example, visualizing a 2D map of the [[MNIST]] images.

>[!warning]
> t-SNE is not meant to be used as a preprocessing stage for an ML model.

- #LDA — Linear discriminant analysis (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`) is a linear **classification** algorithm that, during training, learns the most **discriminative axes between the classes**. Those axes define a hyperplane onto which you can project the data.

	Because the projection keeps classes **as far apart as possible**, LDA is a good technique to reduce dimensionality before running another classification algorithm (unless LDA alone is already sufficient).

- #UMAP — Uniform Manifold Approximation and Projection is another popular technique for visualization. Where t-SNE is better at preserving **local** structure (especially clusters), UMAP tries to preserve **both local and global** structure, and it scales better to large datasets.

>[!note]
> UMAP isn't available in Scikit-Learn, but there's a good implementation in the [`umap-learn`](https://umap-learn.readthedocs.io/) package.

---

![[7-11.png]]
> Reducing the [[Swiss Roll]] to 2D with various techniques:
> -  #MDS flattens it without losing its global curvature
> - #Isomap drops the curvature entirely
> - #t-SNE flattens it while keeping a bit of curvature and amplifies clusters, tearing the roll apart. 
>  
> Whether preserving large-scale structure is good or bad depends on the downstream task.

