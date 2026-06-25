# Ensemble Learning
#ensemble-learning means **aggregating the predictions of a group of predictors** (classifiers or regressors) to get better predictions than the best individual one.

The intuition is the ***wisdom of the crowd***: pose a complex question to thousands of random people and aggregate their answers, and the result often beats an expert's. Same idea with models — many predictors voting together tend to outperform any single one.

A classic example: train many decision tree classifiers, each on a different random subset of the training set, then let them vote — the class with the most votes wins. Such an ensemble of trees is a #random-forests, one of the most powerful ML algorithms available despite its simplicity.

The most popular ensemble methods:
- [[Voting Classifiers]]
- [[Bagging and Pasting]]
- [[Random Forests]]
- [[Boosting]]
- [[Stacking]]

> [!tip] 
> You'll typically reach for ensembles **near the end of a project**, once you already have a few good predictors, to combine them into an even better one.

>[!warning]
> Ensembles need far more compute (training and inference), are more complex to deploy and manage, and their predictions are harder to interpret. But the pros often outweigh the cons.
