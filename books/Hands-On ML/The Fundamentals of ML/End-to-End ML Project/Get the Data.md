# Get the Data
## Download the Data
In typical environments, data is often stored in a **relational database** or another common data store and distributed across multiple tables, documents, or files. Before you access it, you must **first become familiar with the data schema**.

Instead of **manually downloading and decompressing the data**, it is usually better to **automate the process with a function** or a script. This makes it easy to fetch the latest version of the data whenever it changes, either on demand or through a scheduled job.

Let's assume we run our data-loading function and the data is loaded into a #pandas **DataFrame**:

```python
import pandas as pd

def load_data():
	# Connect to the database, download the data, and load it into a DataFrame   
	return df

df = load_data()
```
## Take a Quick Look at the Data Structure
You start by **looking at the top five rows** of data using the DataFrame’s `head()` method.

```python
df.head()
```

The `info()` method is useful to get a quick description of the data, in particular the total **number of rows**, each **attribute’s type**, and the number of **non-null values**:

```python
df.info()
```

Suppose you encounter a **categorical attribute**. You can use the `value_counts()` method to see the categories it contains and how many instances belong to each category.

```python
df["cat_var"].value_counts()
```

The `describe()` method shows a summary of the **numerical attributes**.

```python
df.describe()
```

Another quick way to get a feel of the type of data you are dealing with is to **plot a histogram for each numerical attribute**.

```python
import matplotlib.pyplot as plt

housing_full.hist(bins=50, figsize=(12, 8))
plt.show()

# The number of value ranges can be adjusted using the bins argument (try playing
# with it to see how it affects the histograms)
```

You should now have a better understanding of the kind of data you’re dealing with!
## Create a Test Set
Before exploring the data any further, you should create a test set, set it aside, and **never look at it during development**! Otherwise, you risk introducing [[Data Snooping Bias]].

> [!note]
> [[Scikit-Learn API]] provides a number of splitter classes in the `sklearn.model_selection` package that implement various strategies to split your dataset into a training set and a test set. #sklearn 

### Random Sampling
Creating a test set is theoretically simple; **pick some instances randomly**, typically 20% of the dataset (or less if your dataset is very large), and set them aside:

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2)
```

Well, this works, but it is not perfect: **if you run the program again, it will generate a different test set!** Over time, you (or your machine learning algorithms) will get to see the whole dataset, which is what you want to avoid.
### Reproducible Splits
One solution is to save the test set on the first run and then load it in subsequent runs.  Another option is to **fix the random seed** so that the same split is generated every time.

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
```

While this guarantees reproducibility, it introduces another problem: **the split may change when new data is added to the dataset**.
### Stable Splits Across Dataset Updates
To keep the train/test split stable even after refreshing the dataset, a common approach is to use a unique and immutable **ID for each instance**.

The idea is to compute a hash of each instance's identifier and assign it to the test set if the hash falls within a predefined range (e.g., the lowest 20% of hash values). This ensures that:
- Existing instances always remain in the same set.
- New instances have a chance of being added to the test set.
- No instance that was previously in the training set suddenly appears in the test set.

Unfortunately, not all datasets contain a suitable identifier column.

The simplest alternative is to use the row index as the identifier. However, this only works if:
- New data is always appended to the end of the dataset.
- Existing rows are never deleted.

If these conditions cannot be guaranteed, you should build an identifier from one or more stable features that uniquely identify each instance.
### Stratified Sampling
So far, we have considered purely random sampling methods. This generally works well when the dataset is large enough (especially relative to the number of attributes). However, **with smaller datasets, random sampling may introduce significant [[Sampling Bias]]**.

To reduce this risk, use #stratified-sampling. This method divides the population into homogeneous subgroups, called *strata*, and samples the appropriate proportion of instances from each stratum. This helps ensure that the train and test sets remain representative of the original dataset.

```python
from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(
    df,
    test_size=0.2,
    stratify=df["cat_var"],
    random_state=42
)
```

For example, if 30% of the instances belong to a particular category, stratified sampling will preserve approximately the same proportion in both the training and test sets.

>[!warning] 
>Test set generation is often overlooked, but it is a **critical step in any machine learning project**. A properly constructed test set provides an unbiased estimate of how well the final model will perform on unseen data, while poor test set design can lead to misleading evaluation results.