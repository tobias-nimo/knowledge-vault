# Multioutput Classification
#multioutput classification **generalizes [[Multilabel Classification|multilabel classification]] so that each label can itself be multiclass** — i.e. take more than two possible values.

To illustrate, let's build a system that **removes noise from images**. It takes a noisy digit image as input and (hopefully) outputs a clean one, represented as an array of pixel intensities. The output is multilabel (**one label per pixel**) and each label can take many values (**pixel intensity from 0 to 255**) — so this is a multioutput classification system.

First, we create the dataset by adding noise to the [[MNIST]] pixel intensities. The targets are the original, clean images:

```python
import numpy as np
from my_datasets import X_train, X_test # defined in ./performance-measures

rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0, 100, (len(X_train), 784))
noise_test = rng.integers(0, 100, (len(X_test), 784))

X_train_mod = X_train + noise_train # noisy inputs
X_test_mod = X_test + noise_test

y_train_mod = X_train               # clean targets
y_test_mod = X_test
```

![[3-12.png]]
> A noisy input image (left) and the target clean image (right).

Now we train the classifier and let it clean up the first test image:

```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
```

![[3-13.png]]
> The cleaned-up image — close enough to the target!
