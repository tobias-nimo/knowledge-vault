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
	{'preprocessing__geo__n_clusters': [5, 8, 10],
	'random_forest__max_features': [4, 6, 8]},
	{'preprocessing__geo__n_clusters': [10, 15],
	'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(
	full_pipeline, param_grid, cv=3,
	scoring='neg_root_mean_squared_error'
)

grid_search.fit(features, labels)
```
> Notice that you can access the hyperparameters of any estimator inside a pipeline, even if it is nested several levels deep. Scikit-Learn uses double underscores (`__`) to navigate through the pipeline hierarchy.

