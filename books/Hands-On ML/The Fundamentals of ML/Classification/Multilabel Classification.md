# Multilabel Classification
A #multilabel classifier **outputs multiple binary tags per instance**.

Think of a face-recognition classifier trained to recognize Alice, Bob, and Charlie. Shown a picture of Alice and Charlie, it should output `[True, False, True]` — one tag per person, "Alice yes, Bob no, Charlie yes".

Here's a simpler illustration on [[MNIST]]. We build two labels per digit — whether it's *large* (7, 8, or 9) and whether it's *odd* — and stack them into a target array with one row per instance and one column per label:

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from my_datasets import X_train, y_train, some_digit # defined in ./performance-measures

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd] # one column per label

knn_clf = KNeighborsClassifier() # supports multilabel natively (not all classifiers do)
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit]) # array([[False, True]])
# the digit 5 is indeed not large (False) and odd (True) -> correct
```

There are many ways to evaluate a multilabel classifier, and the right metric depends on your project. One approach is to compute a binary metric (like the #F1-score) **for each individual label, then average the scores**:

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro") # 0.976410265560605
```

This assumes **all labels are equally important**, which may not hold — e.g. if you have many more pictures of Alice than of Bob or Charlie. One option is to weight each label by its **support** (the number of instances with that label) by setting `average="weighted"`.
## Chain Classifiers
If you want to use a classifier that doesn't natively support multilabel classification (such as `SVC`), one strategy is to train one model per label. The problem: this can't capture **dependencies between labels**.

The fix is to **organize the models in a chain**: when a model makes a prediction, it uses the input features plus all predictions from the models before it in the chain. Scikit-Learn's `ClassifierChain` does exactly this.

```python
from sklearn.svm import SVC
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000]) # first 2k images to speed things up

chain_clf.predict([some_digit]) # array([[0., 1.]]) -> correct
```

>[!note] How the chain is trained
>By default, `ClassifierChain` trains each model on the **true labels** of the models before it. If you set the `cv` hyperparameter, it instead uses cross-validation to produce **clean out-of-sample predictions** from each trained model, and feeds those to the later models in the chain. 
>
>The **order of the classifiers** in the chain may affect final performance.