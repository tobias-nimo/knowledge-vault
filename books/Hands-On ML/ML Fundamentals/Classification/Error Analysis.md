# Error Analysis
Let's suppose we've shortlisted a promising classifier for the [[MNIST]] dataset:

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from my_datasets import X_train, y_train # defined in ./performance-measures

sgd_clf = SGDClassifier(random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

cross_val_score(
	sgd_clf, X_train_scaled, y_train,
	cv=3, scoring="accuracy" # since classes are balanced, accuracy is fine
)
# array([0.8983, 0.891 , 0.9018])
```

One way to improve it is #error-analysis — **analyzing the types of errors the model makes**. Understanding the model's most common mistakes lets you target fixes, such as #feature-engineering or just adding more examples of the most misclassified instances.

## Looking at the confusion matrix
First get clean out-of-sample predictions with `cross_val_predict()`, then plot the matrix with `ConfusionMatrixDisplay.from_predictions()`:

```python
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show() # figure 3-9 (left)
```

![[3-9.png]]
> **Left:** raw confusion matrix. **Right:** the same matrix normalized by row.

Most images sit on the **main diagonal**, meaning they were classified correctly. But the cell for class 5 looks a bit darker than the others — this could mean the model makes more errors on 5s, or simply that there are fewer 5s in the dataset. To remove that ambiguity, we can **normalize each row by its sum** (the total number of images of that true class):

```python
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show() # figure 3-9 (right)
```

Now we can read it directly: only 82% of 5s were classified correctly, and the most common mistake was misclassifying them as 8s (10% of all 5s). Note that only 2% of 8s got misclassified as 5s.

Many digits get misclassified as 8s, but that isn't obvious from the matrix above because the correct (diagonal) predictions dominate. To highlight the mistakes, **put zero weight on correct predictions**:

```python
sample_weight = (y_train_pred != y_train) # 0 for correct predictions, 1 for errors
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show() # figure 3-10 (left)
```

![[3-10.png]]
> Confusion matrix with **errors only**. **Left:** normalized by row. **Right:** normalized by column.

The column for class 8 is now very bright, confirming that many images get misclassified as 8s — in fact it's the most common misclassification for almost every class.

>[!warning] Read these percentages carefully — the correct predictions were excluded.
>The 36% in row #7, column #9 does not mean 36% of all 7s were misclassified as 9s. It means **36% of the *errors* made on 7s were misclassifications as 9s**. In reality only 3% of all 7s were misclassified as 9s.

You can also **normalize by column** instead of by row. For example, 56% of misclassified 7s are actually 9s.

```python
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="pred", values_format=".0%")
plt.show() # figure 3-10 (right)
```

## Analyzing individual errors
Plotting individual misclassified examples can reveal *why* the model fails.

```python
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)] # true 3s, pred 3
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)] # true 3s, pred 5
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)] # true 5s, pred 3
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)] # true 5s, pred 5
[...] # plot all four blocks in a confusion matrix style
```

![[3-11.png]]
> Some images of 3s and 5s organized like a confusion matrix.

Most of the misclassified digits look like obvious errors. This is probably because we used a plain `SGDClassifier`, a **linear model**: it assigns a weight per class to each pixel and sums the weighted pixel intensities to score a new image. Since 3s and 5s differ by only a few pixels, this model confuses them easily.