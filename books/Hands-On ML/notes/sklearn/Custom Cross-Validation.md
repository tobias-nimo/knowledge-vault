# Custom Cross-Validation
#sklearn 

Occasionally you need more control over the #cross-validation process than Scikit-Learn offers off the shelf. In those cases you can **implement it yourself**. The code below does roughly the same thing as `cross_val_score()`:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from my_datasets import X_train, y_train
from my_predictors import my_model

skfolds = StratifiedKFold(n_splits=3) # add shuffle=True if not already shuffled

for train_index, test_index in skfolds.split(X_train, y_train):
    clone_ = clone(my_model)
    
    X_train_folds = X_train[train_index]
    y_train_folds = y_train[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train[test_index]

    clone_.fit(X_train_folds, y_train_folds)
    y_pred = clone_.predict(X_test_fold)
    
    # change the performance metric based on your specific use case
    n_correct = sum(y_pred == y_test_fold)
    accuracy = n_correct / len(y_pred)
    print(accuracy)
```

`StratifiedKFold` performs #stratified-sampling to produce folds with a representative ratio of each class. At each iteration the code clones the classifier, trains the clone on the training folds, predicts on the test fold, and outputs the ratio of correct predictions.