# Unsupervised Learning
#unsupervised-learning means learning from **unlabeled data**: you have the input features $X$, but no labels $y$, so the algorithm must find structure in the data on its own.

Besides [[Dimensionality Reduction|dimensionality reduction]], the main **unsupervised tasks** are:

- **Clustering** — group similar instances together into *clusters*. Useful for data analysis, customer segmentation, recommender systems, search engines, semi-supervised learning, and more.
- **Anomaly detection** — learn what "normal" data looks like, then use that to detect abnormal instances (*anomalies* or *outliers*; normal instances are called *inliers*). Useful for fraud detection, detecting defective products, identifying new trends in time series, or cleaning a dataset before training another model.
- **Density estimation** — estimate the **probability density function (PDF)** of the random process that generated the dataset. Commonly used for anomaly detection (instances in very low-density regions are likely anomalies), data analysis, and visualization.

---

Notes on unsupervised learning techniques:

- [[k-Means]]
- [[DBSCAN]]
- [[Gaussian Mixtures]]

Clustering can also power [[Semi-Supervised Learning|semi-supervised learning]] and [[k-Means#k-Means|non-linear dimensionality reduction]].

> [!note] Cluster
> There is no universal definition of what a cluster is: it really depends on the context, and different algorithms will capture different kinds of clusters. 
> 
> Some algorithms look for instances centered around a particular point, called a centroid. Others look for continuous regions of densely packed instances: these clusters can take on any shape. Some algorithms are hierarchical, looking for clusters of clusters. And the list goes on.