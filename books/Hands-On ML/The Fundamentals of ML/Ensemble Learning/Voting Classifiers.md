# Voting Classifiers
A #voting-classifier aggregates the predictions of several different classifiers and **outputs the class that gets the most votes**.

Suppose you've trained a few diverse models — logistic regression, an SVM, a random forest, a k-nearest neighbors classifier — each reaching about 80% accuracy.

![[6-1.png]]
> Training diverse classifiers on the same training set.

Simply having them vote often produces a classifier that beats every individual one. When the prediction is the plain majority vote, it's called hard voting.

![[6-2.png]]
> A hard voting classifier picks the class with the most votes.

Even if each classifier is a #weak-learner (only slightly better than random guessing), the ensemble can be a #strong-learner (high accuracy) — *provided there are enough of them and they're diverse enough* (they focus on different aspects of the data and make different kinds of errors).

>[!tip] Use independent predictors!
> Ensembles work best when predictors are as **independent** as possible. 
> 
> Get diversity by using **very different algorithms**, **tuning hyperparameters** differently, or **training on different subsets** of the data — anything that makes them make different types of errors.

## Hard voting
Scikit-Learn provides a `VotingClassifier` class that takes a list of name/predictor pairs and behaves like a normal classifier. 

Trying it on the [[Moons|moons dataset]]:

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)
```

Each fitted classifier's accuracy on the test set:

```python
for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))
# lr = 0.864
# rf = 0.896
# svc = 0.896
```

`predict()` performs hard voting — here it predicts class 1 for the first instance because two of three classifiers vote for it:

```python
voting_clf.predict(X_test[:1]) # array([1])
[clf.predict(X_test[:1]) for clf in voting_clf.estimators_] 
# [array([1]), array([1]), array([0])]

voting_clf.score(X_test, y_test) # 0.912 -> the ensemble outperforms all of its members
```

## Soft voting
If every classifier can estimate class probabilities (has a `predict_proba()` method), you can average those probabilities and predict the class with the highest average. This is #soft-voting, and it often beats hard voting because it **gives more weight to confident votes**.

Set `voting="soft"` and make sure all estimators expose probabilities.

```python
# SVC doesn't expose probs by default
# set its probability attribute to True (uses cross-validation, slower) to add predict_proba()
voting_clf.named_estimators["svc"].probability = True

voting_clf.voting = "soft"
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test) # 0.92 -> 92% accuracy just by switching to soft voting
```


>[!warning] Calibration matters
> Soft voting works best when the estimated probabilities are **well-calibrated**. 
> 
> If they aren't, calibrate them with `sklearn.calibration.CalibratedClassifierCV`.
