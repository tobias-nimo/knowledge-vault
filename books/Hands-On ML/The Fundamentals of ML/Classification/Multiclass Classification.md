# Multiclass Classification
Whereas binary classifiers distinguish between two classes, #multiclass classifiers (also called *multinomial classifiers*) can distinguish between **more than two classes**.

Some Scikit-Learn classifiers (e.g., `LogisticRegression`, `RandomForestClassifier`, and `GaussianNB`) **handle multiple classes natively**. Others are **strictly binary** (e.g., `SGDClassifier` and `SVC`). Still, you can combine several binary classifiers to perform multiclass classification using one of two strategies.

## OvA and OvO
The #one-versus-all (OvA) strategy — also called *one-versus-the-rest* (OvR) — trains **one binary classifier per class**. To classify the [[MNIST]] digits into 10 classes, you train a 0-detector, a 1-detector, a 2-detector, and so on. To classify an image, you get the decision score from each classifier and **select the class whose classifier outputs the highest score**.

The #one-versus-one (OvO) strategy trains **a binary classifier for every pair of classes**: one to distinguish 0s and 1s, another for 0s and 2s, another for 1s and 2s, and so on. With $N$ classes you need $N \times (N - 1) / 2$ classifiers — for MNIST that's 45 binary classifiers! To classify an image, you run it through all 45 and **see which class wins the most duels**. The main advantage of OvO is that each classifier only needs to be trained on the part of the training set containing the two classes it must distinguish.

>[!tip] Which strategy to use?
>Some algorithms (such as SVMs) **scale poorly with the size of the training set**. For these, **OvO is preferred** because it's faster to train many classifiers on small training sets than few classifiers on large ones. For most binary classification algorithms, however, OvA is preferred.

Scikit-Learn detects when you use a binary algorithm for a multiclass task and **automatically runs OvA or OvO, depending on the algorithm**.

For example, if we train an SVM model with 10 classes, Scikit-Learn will use **OvO** under the hood and train 45 binary classifiers. The `decision_function()` will return 10 scores per instance — one per class. Each class gets a score equal to the number of won duels ± max 0.33 to break ties, based on the classifier scores.

```python
from sklearn.svm import SVC
from my_datasets import X_train, y_train, some_digit # defined in ./performance-measures

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000]) # train on just 2k images (otherwise it takes very long)

svm_clf.predict([some_digit]) # array(['5'], dtype=object) -> correct

svm_clf.decision_function([some_digit])
# array([[3.79, 0.73, 6.06, 8.3 , -0.29, 9.3 , 1.75, 2.77, 7.21, 4.82]])
#  -> the highest score corresponds to the '5' class
```

Now let's train an `SGDClassifier` instead. Since it's a binary algorithm that doesn't favor small training sets, Scikit-Learn falls back to **OvA**, training just 10 binary classifiers — so `decision_function()` returns one score per class:

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

sgd_clf.predict([some_digit]) # array(['3'], dtype='<U1') -> incorrect! errors happen

sgd_clf.decision_function([some_digit])
# array([[-31893., -34420., -9531., 1824., -22320., -1386., -26189., -16148., -4604., -12051.]])
# -> the highest score corresponds to the '3' class
```

>[!note] In case of a tie...
> The **first class is selected**, unless you set `break_ties=True`, in which case ties are broken using the output of `decision_function()`.

To **force OvO or OvA**, use the `OneVsOneClassifier` or `OneVsRestClassifier` classes. Just pass any classifier to the constructor (it doesn't even have to be binary):

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit]) # array(['5'], dtype='<U1')
len(ovr_clf.estimators_)      # 10 -> one binary classifier per class
```