# Ensemble Learning
#ensemble-learning means **aggregating the predictions of a group of predictors** (classifiers or regressors) to get better predictions than the best individual one.

The intuition is the ***wisdom of the crowd***: pose a complex question to thousands of random people and aggregate their answers, and the result often beats an expert's. Same idea with models — many predictors voting together tend to outperform any single one.

A classic example: train many decision tree classifiers, each on a different random subset of the training set, then let them vote — the class with the most votes wins. Such an ensemble of trees is a #random-forest, one of the most powerful ML algorithms available despite its simplicity.

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

## When to use each method
Ensemble methods are versatile, powerful, and fairly simple to use. They **can overfit if you're not careful**, but that's true of every powerful model.

| Method             | When to use it                                                                                                                      | Example use cases                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| #hard-voting       | Balanced classification dataset with multiple strong but diverse classifiers.                                                       | Spam detection, sentiment analysis, disease classification                          |
| #soft-voting       | Classification dataset with probabilistic models, where confidence scores matter.                                                   | Medical diagnosis, credit risk analysis, fake news detection                        |
| #bagging           | Structured or semi-structured dataset with high variance and overfitting-prone models.                                              | Financial risk modeling, e-commerce recommendation                                  |
| #pasting           | Structured or semi-structured dataset where more independent models are needed.                                                     | Customer segmentation, protein classification                                       |
| #random-forest     | High-dimensional structured datasets with potentially noisy features.                                                               | Customer churn prediction, genetic data analysis, fraud detection                   |
| #extra-trees       | Large structured datasets with many features, where speed is critical and reducing variance is important.                           | Real-time fraud detection, sensor data analysis                                     |
| #adaboost          | Small to medium-sized, low-noise, structured datasets with weak learners (e.g. decision stumps), where interpretability is helpful. | Credit scoring, anomaly detection, predictive maintenance                           |
| #gradient-boosting | Medium to large structured datasets where high predictive power is required, even at the cost of extra tuning.                      | Housing price prediction, risk assessment, demand forecasting                       |
| #HGB               | Large structured datasets where training speed and scalability are key.                                                             | Click-through rate prediction, ranking algorithms, real-time bidding in advertising |
| #stacking          | Complex, high-dimensional datasets where combining multiple diverse models can maximize accuracy.                                   | Recommendation engines, autonomous vehicle decision-making, Kaggle competitions     |

>[!tip] Start here
> Random forests, AdaBoost, GBRT, and HGB are among the **first models you should test** for most ML tasks, particularly with heterogeneous tabular data. They require very little preprocessing, so they're great for getting a prototype up and running quickly.
