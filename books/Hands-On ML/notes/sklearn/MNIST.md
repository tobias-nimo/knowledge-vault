# MNIST
The **MNIST** dataset is a set of **70,000 small images of handwritten digits**, each labeled with the digit it represents.

![[3-2.png]]

You can fetch it from [OpenML.org](https://www.openml.org/) using Scikit-Learn:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target # numpy arrays
```

Each image is **28 × 28 pixels**, so `X` has shape `(70000, 784)`: one feature per pixel, with intensity from 0 (white) to 255 (black). To display a digit, reshape its feature vector and plot it:

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
	image = image_data.reshape(28, 28)
	plt.imshow(image, cmap="binary") # grayscale: 0 white, 255 black
	plt.axis("off")
	
some_digit = X[0]
plot_digit(some_digit)
plt.show()
```

![[3-1.png]]

The images are **clean, centered, and roughly the same size**, so MNIST needs little preprocessing (real-world datasets aren't usually this friendly).

The dataset is **already split** into a training set (first 60,000) and a test set (last 10,000):

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

The training set is also already **shuffled**, which guarantees similar cross-validation folds and avoids problems with algorithms sensitive to instance order.