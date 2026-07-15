# Incremental PCA
One problem with the standard [[PCA]] implementations is that they require the **whole training set to fit in memory**. #incremental-pca instead lets you split the training set into **mini-batches** and feed them in one at a time. This is useful for large training sets, and for applying PCA online (i.e. on the fly, as new instances arrive).

The following code splits [[MNIST]] into 100 mini-batches and feeds one at a time. Note that you must call `partial_fit()` on each mini-batch, rather than `fit()` on the whole set:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]
```

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches): # MNIST, see [[PCA]]
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

## Using `memmap`
Alternatively, NumPy's `memmap` class lets you manipulate a large array stored in a binary file on disk **as if it were entirely in memory**, loading only the data it needs, when it needs it.

First create the memory-mapped file and copy the data into it, then call `flush()` to make sure any data still in the cache gets saved to disk.

```python
filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)

# in real life X_train wouldn't fit in memory, so you'd write it chunk by chunk; could be a loop, saving the data chunk by chunk
X_mmap[:] = X_train

X_mmap.flush()
```

Since incremental PCA only uses a small part of the array at any given time, memory usage stays under control. This lets you call the usual `fit()` instead of `partial_fit()`, which is quite convenient:

```python
# only the raw binary data is saved to disk, so you must specify the dtype and shape when loading it; if you omit the shape, np.memmap() returns a 1D array.
X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)

batch_size = X_mmap.shape[0] // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)
```