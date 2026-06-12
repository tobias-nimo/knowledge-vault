# Testing and Validating
The only reliable way to determine how well a model will generalize to unseen data is to evaluate it on new instances. To do this, the available data is typically split into two sets: a **training set** and a **test set**. The model is trained on the training set and then evaluated on the test set.

The error made on previously unseen instances is known as the **generalization error** (or **out-of-sample error**). By measuring the model's performance on the test set, we obtain an estimate of this error, which indicates how well the model is expected to perform on new data.

If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the training data.

>It is common to use 80% of the data for training and hold out 20% for testing. However, this depends on the size of the dataset.
## Hyperparameter Tuning and Model Selection
When comparing models or tuning hyperparameters, repeatedly evaluating performance on the test set can lead to overfitting to that specific set. As a result, the measured test error may be overly optimistic and fail to reflect performance on truly unseen data.
![[1-25.png]]
To avoid this issue, a **validation set** is carved out of the training data. Candidate models are trained on the remaining data and evaluated on the validation set to select the best one. The chosen model is then retrained on the full training set and finally evaluated on the test set to estimate its generalization error.
## Data Mismatch
Training data may not be fully representative of real-world production data. To diagnose whether poor performance is caused by **overfitting** or **data mismatch**, use a **train-dev set** drawn from the same distribution as the training data. If performance is poor on the train-dev set, the model is likely overfitting; if it performs well on the train-dev set but poorly on the validation set (or **dev set**), the issue is likely data mismatch. Validation and test sets should always be representative of production data.
![[1-26.png]]