# Scikit-Learn Imputers
#sklearn 

Imputers are used to **replace missing values** in a dataset before training a model.
## Basic Usage

```python
from sklearn.impute import SimpleImputer

# Create an imputer
imputer = SimpleImputer(strategy="median") 

# Learn the median of each numerical feature
imputer.fit(features_num) 

# Replace missing values using the learned medians  
X = imputer.transform(features_num)
```

Other available strategies are: 
- the mean value (`strategy="mean"`), 
- the most frequent value (`strategy="most_frequent"`), 
- or a constant value (`strategy="constant", fill_value=...`). 

Note that the last two strategies support non-numerical data. 
## Advanced Imputers
There are also more powerful imputers available in the `sklearn.impute` package (both for numerical features only):
- `KNNImputer` replaces each missing value with the mean of the **k-nearest neighbors**’ values for that feature. The distance is based on all the available features.
- `IterativeImputer` trains a **regression model** per feature to predict the missing values based on all the other available features. It then trains the model again on the updated data, and repeats the process several times, improving the models and the replacement values at each iteration.