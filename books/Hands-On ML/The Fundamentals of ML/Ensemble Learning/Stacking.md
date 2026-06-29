# Stacking
#stacking replaces the trivial aggregation function of an ensemble (like hard voting) with a model trained to combine the predictions of all the base predictors.

The idea is simple: instead of averaging or voting, you feed the base predictors' outputs into a final model — called a **blender** — that learns how to best combine them into the final prediction.

![[6-11.png]]

To train the blender you build a **blending training set** whose input features are the base predictors' outputs and whose targets are copied from the original training set. Regardless of how many features the original data has, this set contains **one input feature per base predictor**.

![[6-12.png]]

>[!warning] Use out-of-sample predictions
> The blender must be trained on **out-of-sample predictions** — predictions each base predictor makes on instances it did *not* see during its own training. 
> 
> If you trained the blender on in-sample predictions, the base models would look artificially good and the blender would learn to over-trust them, hurting generalization.

You can also stack **several blenders** into a layer (e.g. one linear regression, one random forest) and add another blender on top of them to produce the final prediction. This *multilayer* stacking can squeeze out a bit more performance, at the cost of more training time and system complexity.

![[6-13.png]]

## In Scikit-Learn
Scikit-Learn provides `StackingClassifier` and `StackingRegressor`. 

Let's stack three base predictors to fit the [[Moons|moons dataset]]:

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5  # number of cross-validation folds used to build the blending set
)
stacking_clf.fit(X_train, y_train)
stacking_clf.score(X_test, y_test) # 0.928

# we got 92% with soft-voting; whether the gain is worth the extra complexity and compute depends on your use case.
```
