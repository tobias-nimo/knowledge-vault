# Fine-Tune Your Models
Let’s assume that you have a shortlist of promising models. The next step is to **fine-tune their hyperparameters** in order to improve performance.

One approach is to adjust hyperparameters manually and evaluate the model repeatedly. However, this can be tedious and may prevent you from exploring many combinations.
## Grid Search
Instead, you can use #grid-search: given a set of hyperparameter values, uses #cross-validation to evaluate each combination.

For example, the following code **searches for the best combination of hyperparameters** for a random forest model:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from my_pipelines import preprocessing # defined in ./prepare-the-data
from my_datafranes import features, labels # defined in ./prepare-the-data

full_pipeline = Pipeline([
	("preprocessing", preprocessing),
	("random_forest", RandomForestRegressor(random_state=42)),
])

param_grid = [
	{
	'preprocessing__geo__n_clusters': [5, 8, 10],
	'random_forest__max_features': [4, 6, 8]
	},
	{
	'preprocessing__geo__n_clusters': [10, 15],
	'random_forest__max_features': [6, 8, 10]
	},
]

# 'preprocessing__geo__n_clusters'
# "preprocessing" pipeline -> "geo"` transformer -> `n_clusters` hyperparameter 

grid_search = GridSearchCV(
	full_pipeline, param_grid, cv=3,
	scoring='neg_root_mean_squared_error'
)

grid_search.fit(features, labels)
```

>[!note] 
>Notice that you can **access the hyperparameters of any estimator inside a pipeline**, even if it is nested several levels deep. Scikit-Learn uses double underscores (`__`) to navigate through the pipeline hierarchy.

> [!tip] **If fitting the pipeline transformers is computationally expensive**...
>  You can set the pipeline’s `memory` parameter to the path of a **caching directory**: when you first fit the pipeline, Scikit-Learn will save the fitted transformers to this directory. If you then fit the pipeline again with the same hyperparameters, Scikit-Learn will just load the cached transformers.

In total the grid search will explore 3×3 + 2×3 = 15 combinations of hyperparameter values, and it will train the pipeline 3 times per combination, since we are using 3-fold cross validation. This means there will be a grand total of 15 × 3 = 45 rounds of training! **It may take a while**...

Once the search is complete, you can **get the best combination of parameters** like this:

```python
grid_search.best_params_
# {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}

# mmm... since `15` is the largest value tested for `n_clusters`, it may be worth running another search with larger values to see whether performance continues to improve.
```
By default, `GridSearchCV` uses `refit=True`: after identifying the best hyperparameter combination, it **retrains the model on the entire training set**. You can access the best model like this:

```python
best_model = grid_search.best_estimator_
```

You can access the **evaluation scores** like this:

```python
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
# [...] # change column names to fit and show rmse = -score
cv_res.head()
```

```
n_clusters max_features split0 split1 split2 mean_test_rmse
15 6 42725 43708 44335 43590
15 8 43486 43820 44900 44069
10 4 43798 44036 44961 44265
10 6 43710 44163 44967 44280
10 6 43710 44163 44967 44280
```
## Randomized Search
#randomized-search works much the same way as grid search, but instead of evaluating very possible combination, it **evaluates a fixed number of randomly selected combinations**. At each iteration, it samples a value for each hyperparameter and evaluates the resulting model using #cross-validation.

For each hyperparameter, you must provide either a list of possible values, or a probability distribution:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint 

param_distributions = {
	"preprocessing__geo__n_clusters": randint(3, 50),
	"random_forest__max_features": randint(2, 20), 
} 

rnd_search = RandomizedSearchCV(
	full_pipeline, param_distributions=param_distributions, cv=3,
	scoring="neg_root_mean_squared_error", random_state=42, n_iter=10
)

rnd_search.fit(features, labels)
```

> [!tip] As a **rule of thumb**:
> - Use #grid-search when you have a small number of hyperparameters and a limited set of candidate values.
> - Use #randomized-search when the search space is large or when you have a limited computational budget.
## Analyzing the Best Models
You will often **gain good insights on the problem** by inspecting the best models. 

For example, some models can indicate the **relative importance of each attribute** for making accurate predictions:

```python
final_model = rnd_search.best_estimator_ # includes preprocessing (`RandomForestRegressor`)
feature_importances = final_model["random_forest"].feature_importances_
features_names = final_model["preprocessing"].get_feature_names_out()

sorted(zip(feature_importances, features_names), reverse=True)
```

```
[(np.float64(0.18599734460509476), 'log__var1'),
(np.float64(0.07338850855844489), 'cat__var2'),
(np.float64(0.06556941990883976), 'var3__ratio'),
[...]
(np.float64(0.0004325970342247361), 'cat__var4'),
(np.float64(3.0190221102670295e-05), 'cat__var5')]
```
With this information, you may want to **try dropping some of the less useful features**.
### Error Analysis
You should also **look at the specific errors** that your system makes, then try to understand why it makes them and what could fix the problem: adding extra features or getting rid of uninformative ones, cleaning up outliers, etc.
### Fairness Analysis
You should also evaluate fairness, ensuring that the model performs well not only on average but **across different groups** (e.g., urban vs. rural, rich vs. poor, etc.). This requires a **bias analysis on validation-set subsets**. If the model performs poorly for a particular group, it should be improved before deployment or restricted from making predictions for that group to avoid potential harm.
## Evaluate on the Test Set
Once you have finished tweaking your models and have a system that performs sufficiently well, **retrieve the test set and use your full pipeline** (`final_model`) to transform the data and generate predictions. Then, evaluate the model's performance on these predictions.

```python
X_test = test_set.drop("target", axis=1)
y_test = test_set["target"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse) # 41445.533268606625
```

A single test-set score provides only a point estimate of the model's generalization error. To **measure the uncertainty of this estimate**, you can compute a #confidence-interval (e.g., 95%) using techniques such as #bootstrapping.

```python
from scipy import stats

def rmse(squared_errors):
	return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
	confidence_level=confidence, random_state=42
)

rmse_lower, rmse_upper = boot_result.confidence_interval # 39,521 to 43,702,
```

If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation. That’s because your system ends up fine-tuned to perform well on the validation data.

>[!warning] You must **resist the temptation to tweak the hyperparameters** to make the numbers look good on the test set; the improvements would be unlikely to generalize to new data.