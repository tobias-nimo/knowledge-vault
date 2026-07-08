# Semi-Supervised Learning
#semi-supervised-learning applies when you have **many unlabeled instances but only a few labeled ones**. Clustering can help in two ways: it can **identify the most representative instances to label** (e.g. those closest to the cluster centroids), and it can **propagate those labels** to the remaining instances in the same cluster.

## Case Study: The digits dataset
Let's try these ideas on the *digits* dataset: a simple [[MNIST]]-like dataset with 1,797 grayscale 8 × 8 images of the digits 0 to 9.

```python
from sklearn.datasets import load_digits

X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]
```

Suppose the dataset is unlabeled and we only have enough time to manually label 50 instances. A first idea is to pick those 50 instances at random.

Training a logistic regression on those 50 labeled instances gives a disappointing baseline:

```python
from sklearn.linear_model import LogisticRegression

n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test) # 0.7581863979848866
```

Just 75.8% accuracy — training on the *full* labeled training set would reach about 90.9%. Let's see how close we can get without labeling more than 50 instances.

### Labeling Representative Instances
Since labeling is often expensive and time-consuming — especially when it must be done manually by experts — it's a good idea to **label representative instances rather than random ones**.

Let's cluster the training set into $k = 50$ clusters and, **for each cluster, find the image closest to the centroid** — the representative images:

```python
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train) # distance of each instance to each centroid
representative_digit_idx = X_digits_dist.argmin(axis=0)
X_representative_digits = X_train[representative_digit_idx]
```

![[8-12.png]]
> The fifty most representative instances (one per cluster).

Now we look at each image and **manually label it**:

```python
y_representative_digits = np.array([8, 0, 1, 3, 6, 7, 5, 4, 2, 8, ..., 6, 4])
```

We still have only 50 labeled instances, but instead of random instances, each is a representative of its cluster:

```python
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test) # 0.8337531486146096
```

We jumped from 75.8% to 83.4% accuracy with the same labeling budget.

### Label Propagation
We can go a step further: **propagate each representative's label to all the other instances in its cluster**. This is called #label-propagation:

```python
y_train_propagated = np.empty(len(X_train), dtype=np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test) # 0.8690176322418136
```

Another significant boost: 86.9%.

We can do even better by **dropping the outliers**: ignoring the 50% of instances farthest from their cluster center, since instances near a cluster boundary are more likely to get the wrong propagated label. 

The code computes each instance's distance to its closest centroid, then for each cluster marks the 50% largest distances with –1 and filters them out:

```python
percentile_closest = 50

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test) # 0.8841309823677582
```

Starting with just 50 manually labeled instances (about five per class), we reach 88.4% accuracy, surprisingly close to the 90.9% obtained with the fully labeled dataset. This works partly because we dropped some outliers, and partly because the propagated labels are actually very good — their accuracy is about 98.9%:

```python
(y_train_partially_propagated == y_train[partially_propagated]).mean() # 0.9887798036465638
```

>[!note] Automatic Label Propagation in Scikit-Learn
> The `sklearn.semi_supervised` package offers classes that propagate labels automatically:
> - `LabelSpreading` and `LabelPropagation` build a **similarity matrix** between all instances and iteratively propagate labels from labeled instances to similar unlabeled ones.
> - `SelfTrainingClassifier` takes a base classifier (e.g. `RandomForestClassifier`), trains it on the labeled instances, predicts labels for the unlabeled ones, adds the **most confident predictions** to the training set, and repeats until it can't add any more.
>
> These techniques are not magic bullets, but they can occasionally give your model a little boost.

## Active Learning
To keep improving the model and the training set, the next step could be a few rounds of #active-learning: a **human expert interacts with the learning algorithm**, labeling specific instances when the algorithm requests them. The most common strategy is **uncertainty sampling**:

1. Train the model on the labels gathered so far, and make predictions on all the unlabeled instances.
2. Give the instances the model is **most uncertain about** (lowest estimated probability) to the expert for labeling.
3. Iterate until the performance improvement stops being worth the labeling effort.

Other strategies include labeling the instances that would cause the largest model change, the largest drop in validation error, or the instances that different models disagree on (e.g. an SVM and a random forest).
