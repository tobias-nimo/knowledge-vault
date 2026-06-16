# Performance Measures
In these notes we'll **explore different performance measures for evaluating a classifier** — a task that's often trickier than evaluating a regressor.

To do this, we'll use the [[MNIST]] dataset and a **binary classifier**:
```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target # numpy arrays
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5') # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')
```

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

some_digit = X[0] # actual 5
sgd_clf.predict([some_digit]) # array([True]) -> correct prediction
```

>Stochastic gradient descent ( #SGD ) classifier is always a good place to start. It is a simple algorithm and it handles very large datasets efficiently.
## Accuracy
Let's use #cross-validation to evaluate our classifier measuring #accuracy:
```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.95035, 0.96035, 0.9604])
```
Above 95% accuracy on every fold — looks amazing, right? Before getting too excited, let's **compare it with a dummy classifier** that always predicts the most frequent class (here the negative class, *non-5*):
```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
any(dummy_clf.predict(X_train)) # False: no 5s detected

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.90965, 0.90965, 0.90965])
```
Over 90% accuracy! This happens simply because only about 10% of the images are 5s, so always guessing *not a 5* is right about 90% of the time.

>[!warning] This is why **accuracy is generally not the preferred performance measure for classifiers**, especially with skewed datasets (when some classes are much more frequent than others).
## Confusion Matrix
 To compute the #confusion-matrix, you **first need a set of predictions** that can be compared against the actual targets. 
 
 The `cross_val_predict()` function gives you exactly that: a **clean prediction for every instance** in the training set. By clean I mean out-of-sample — each prediction is made by a model that never saw that instance during training.
```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```
Now you are ready to get the confusion matrix using the `confusion_matrix()` function:
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
cm
```

```
array(
	[[53892, 687],
	[ 1891, 3530]]
)
```

If you are confused about the confusion matrix, the figure below illustrates its main components in the context of this example.
![[3-3.png]]
## Precision and Recall
#precision is the accuracy of the positive predictions:
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$
- $\text{TP}$ is the number of true positives
- $\text{FP}$ is the number of false positives.

#recall is the ratio of positive instances that are correctly detected by the classifier:
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$
- $\text{FN}$ is the number of false negatives.

It is often convenient to **combine precision and recall** into a single metric called the #F1-score, especially when you need a single metric to compare two classifiers.
$$
\text{F}_1
=
\frac{2}{\frac{1}{\text{precision}}+\frac{1}{\text{recall}}}
=
2 \cdot \frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}
=
\frac{\text{TP}}{\text{TP}+\frac{\text{FN}+\text{FP}}{2}}
$$
The $\text{F}_1$ score is computed with the harmonic mean, so it **favors classifiers that have similar precision and recall**. 

Scikit-Learn provides functions to compute these metrics:
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred) # == 3530 / (687 + 3530) = 0.8370
recall_score(y_train_5, y_train_pred) # == 3530 / (1891 + 3530) = 0.6511
f1_score(y_train_5, y_train_pred) # 0.7325
```
## Precision/Recall Curve
In some contexts you mostly care about precision, and in other contexts you really care about recall. Unfortunately, you can’t have it both ways: **increasing precision reduces recall, and vice versa**. This is called the #precision-recall-trade-off.

To understand this trade-off, let's look at **how most classifiers make their decisions**. For each instance, they compute a score using a decision function. If that score is greater than a threshold, the instance is assigned to the positive class; otherwise it's assigned to the negative class.
![[3-4.png]]
Instead of `predict()`, you can call `decision_function()` to get each instance's score, then threshold it yourself:
```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores # array([2164.22030239])

threshold = 0
(y_scores > threshold) # array([True])  -- same as predict() (default threshold is 0)

threshold = 3000
(y_scores > threshold) # array([False]) -- raising the threshold misses the 5 -> lower recall
```
**How do you pick a threshold?** Get clean decision scores for every instance with `cross_val_predict(..., method="decision_function")`, then feed them to `precision_recall_curve()`:
```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```
Now you can **plot precision and recall as functions of the threshold** and inspect what trade-off suits your project best:
```python
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
plt.show()
```
![[3-5.png]]
Another way to choose a trade-off is to **plot precision directly against recall** ( #PR curve):
```python
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.show()
```
![[3-6.png]]
Suppose you **aim for 90% precision**. Search for the lowest threshold that achieves it using `argmax()`:
```python
idx_for_90_precision = (precisions >= 0.90).argmax() # first index of the max value (here, the first True)
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision # 3370.01949

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

precision_score(y_train_5, y_train_pred_90) # 0.9000
recall_score(y_train_5, y_train_pred_90)    # 0.4800
```
You now have a 90% precision classifier! It's actually **easy to reach almost any precision**: just set a high enough threshold. But a high-precision classifier isn't very useful if its recall is too low.

>[!tip] If someone says "Let's reach 99% precision", you should ask, "**At what recall?**"

> #sklearn  offers two classes that make adjusting the threshold easier:
> - [`FixedThresholdClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.FixedThresholdClassifier.html) wraps a binary classifier and lets you set the threshold manually.
> - [`TunedThresholdClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) uses k-fold cross validation to automatically find the optimal threshold for a given metric (by default, it tries to find the threshold that maximizes the model’s balanced accuracy — the average of each class’s recall).
## ROC Curve
The #ROC (receiver operating characteristic) curve is another **common tool for binary classifiers**. It plots the **true positive rate** (another name for #recall or *sensitivity*) against the **false positive rate** (another name for #fall-out).
$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$
$$
\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$
$$\text{FPR}=1 - \text{TNR}$$
- $\text{TN}$ is the number of true negatives.
- $\text{FPR}$ is the ratio of negative instances incorrectly classified as positive.
- $\text{TPR}$ is the ratio of positive instances correctly classified as positive.
- $\text{TNR}$ (also called *specificity*) is the ratio of negative instances that are correctly
classified as negative.

Use `roc_curve()` to compute the TPR and FPR for various thresholds, then plot them:
```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# thresholds are in decreasing order here, so we use <= to find the 90% precision point
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.show()
```
![[3-7.png]]
The dotted line is the **ROC curve of a purely random classifier**; a good classifier stays as far from it as possible (toward the top-left corner). Once again there's a trade-off: **the higher the recall (TPR), the more false positives (FPR)**. 

One way to compare classifiers is the **area under the curve** ( #AUC ): a perfect classifier has ROC AUC equal to 1, a purely random one has 0.5.
```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores) # 0.9604
```
## Comparing Classifiers

>[!tip] ROC curve or PR curve?
>As a rule of thumb, prefer the **PR curve whenever the positive class is rare** or when you care more about false positives than false negatives. Otherwise, use the ROC curve.

This is exactly our case: the 5s are rare (~10% of the data), so we'll compare classifiers with the PR curve. The reason is that the FPR (used by the ROC curve) divides by the huge pool of negatives, so even a fair number of false positives barely moves it — which makes the ROC AUC (0.96) look flattering. The PR curve ignores the abundant true negatives and looks only at the positive class, exposing how the classifier actually does on the 5s.

Let's compare the `SGDClassifier` against a `RandomForestClassifier`. It has no `decision_function()` method, but it does have `predict_proba()`, which returns estimated class probabilities — we can use the probability of the positive class as the score:
```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_probas_forest[:2]
```

```
array(
	[[0.11, 0.89],   89% probability the 1st image is a 5
    [0.99, 0.01]]    99% probability the 2nd image is NOT a 5
)
```

The second column holds the positive-class probabilities, so we use it as the score for the PR curve:
```python
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.show()
```
![[3-8.png]]
The random forest's PR curve comes much closer to the **top-right corner**, so it's clearly superior to the SGD classifier.

Its $\text{F}_1$ and ROC AUC scores are also significantly better:
```python
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5 # positive proba >= 50%

f1_score(y_train_5, y_train_pred_forest)   # 0.9275
roc_auc_score(y_train_5, y_scores_forest)  # 0.9983
# precision ~99.0% and recall ~87.3%
```
