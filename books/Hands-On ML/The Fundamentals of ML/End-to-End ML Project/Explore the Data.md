# Explore and Visualize the Data to Gain Insights
First, make sure you have put the test set aside and you are **only exploring the training set**. 

**Make a copy** of the original training set so you can safely experiment and revert changes if needed.
```python
# Create a copy of the training set
exploration_set = train_set.copy()

# If the training set is large, sample a smaller set for faster analysis
exploration_set = train_set.sample(frac=0.1, random_state=42)
```
## Visualizing Geographical Data
If the dataset contains geographical features such as **latitude and longitude**, creating a scatter plot is a good way to visualize the data.
```python
exploration_set.plot(
	kind="scatter", x="longitude", y="latitude",
	alpha=0.2, grid=True,
)
plt.show()
```
![[2-12.png]]
Our brains are very good at spotting patterns in pictures, but you may need to **play around with visualization parameters** to make the patterns stand out...
```python
exploration_set.plot(
	kind="scatter", x="longitude", y="latitude",
	s=exploration_set["var2"] / 100, label="var2",
	c="var1", cmap="jet", colorbar=True,
	grid=True, legend=True, sharex=False, figsize=(10, 7)
)
plt.show()
```
![[2-13.png]]
## Look for Correlations
If the dataset is not too large, you can easily compute the standard correlation coefficient between every pair of numerical attributes using the `corr()` method:
```python
corr_matrix = exploration_set.corr(numeric_only=True)
```
Now you can look at **how much each attribute correlates with the target value**:
```python
corr_matrix["target"].sort_values(ascending=False)
```
The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation. When the coefficient is close to –1, it means that there is a strong negative correlation. Finally, coefficients close to 0 mean that there is no linear correlation.

Another way to check for correlation between attributes is to use the Pandas `scatter_matrix()` function, which plots every numerical attribute against every other numerical attribute.
```python
from pandas.plotting import scatter_matrix

attributes = ["target", "var1", "var2", "var3"] # only numerical features
scatter_matrix(exploration_set[attributes], figsize=(12, 8))
plt.show()
```
Looking at the **correlation scatterplots**, you may identify attributes that are strongly correlated with the target variable and are therefore **promising predictors**.

> Note that the correlation coefficient only measures linear correlations (“as $x$ goes up, $y$ generally goes up/down”). It may completely miss out on nonlinear relationships (e.g., “as $x$ approaches 0, $y$ generally goes up”).
## Experiment with Attribute Combinations
One last thing you may want to do before preparing the data for machine learning algorithms is to **try out new features by combining existing attributes**.
```python
exploration_set["var1_ratio"] = exploration_set["var1"] / exploration_set["var3"]
exploration_set["var2_per_var3"] = exploration_set["var2"] / exploration_set["var3"]
```
These engineered features may capture relationships that are not apparent in the original attributes. After creating them, recompute the correlation matrix to **check whether they are more strongly correlated** with the target variable than the original features.
```python
corr_matrix = exploration_set.corr(numeric_only=True)
corr_matrix["target"].sort_values(ascending=False)
```

> When creating new combined features, make sure they are not too linearly correlated with existing features: **collinearity can cause issues with some models**, such as linear regression. In particular, avoid simple weighted sums of existing features.

