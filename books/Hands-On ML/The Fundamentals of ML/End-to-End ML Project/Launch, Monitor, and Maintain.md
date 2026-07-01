# Launch, Monitor, and Maintain your System

## Deployment
First, you need to **save the model** you trained. You can use `joblib` for this:

```python
# in your lab environment...

import joblib
joblib.dump(final_model, "my_model.pkl")
```

Now, you have to transfer the file to your production environment, and load it. 

The simplest deployment is to **load the model on server startup** and call its `predict()` method directly from your application code.

```python
# in your app code...

import joblib
# [...] # import KMeans, BaseEstimator, TransformerMixin, rbf_kernel, etc.

# any custom transformers, functions, or classes used during training must be available
def column_ratio(X): [...]
def ratio_name(function_transformer, feature_names_in): [...]
class ClusterSimilarity(BaseEstimator, TransformerMixin): [...]

# load the model
final_model_reloaded = joblib.load("my_model.pkl")

# make predictions on new data
new_data = [...]
predictions = final_model_reloaded.predict(new_data)
```

A more flexible option is to wrap the model in a dedicated **web service** queried through a REST-API.

![[2-21.png]]

Another popular strategy is to **deploy to the cloud** (e.g., Google's Vertex-AI). This gives you a web service that handles load balancing and scaling for you.

## Monitoring
Deployment is not the end of the story: you need **monitoring code that checks live performance** at regular intervals and triggers alerts when it drops. Performance can degrade suddenly (e.g., a infrastructure failures or broken data pipelines) or slowly and unnoticed, often due to #data-drift: real-world data gradually changes and becomes different from the data used during training.

How you monitor performance depends on the task:
- sometimes it can be inferred from **business metrics**
- other times it requires **human evaluation**.

## Maintenance
If the data keeps evolving, you must **update your datasets and retrain regularly**, automating the process as much as possible.

You should also **monitor input data quality**, since a poor-quality signal (e.g., a malfunctioning sensor, or another team's stale output) can degrade performance slowly. Trigger alerts if more inputs are missing a feature, the mean or standard deviation drifts too far from the training set, or a categorical feature gains new categories.

Finally, **keep backups** of every model and dataset version, with the tooling to **roll back quickly** if a new model or dataset fails. Backups also let you compare new models against previous ones and evaluate any model against any previous dataset.

> [!note] ML Operations
> As you can see, ML involves a lot of infrastructure. This broad topic is called #MLOps.



