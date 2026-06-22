# Iris Dataset
The **iris dataset** is a famous dataset containing the **sepal and petal length and width of 150 iris flowers** of three species: *Iris setosa*, *Iris versicolor*, and *Iris virginica*.

![[4-23.png]]

You can load it from #sklearn with `load_iris`:

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True) # returns pandas DataFrame
list(iris)
# ['data', 'target', 'frame', 'target_names', 'feature_names', ...]
```

The four features live in `iris.data`:

```python
iris.data.head(3)
```

```
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
```

The labels are in `iris.target`, encoded as integers (0, 1, 2), with `iris.target_names` mapping each integer to its species name:

```python
iris.target.head(3)  # note that the instances are not shuffled
# 0    0
# 1    0
# 2    0

iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

Because it's small, clean, and only mildly separable, the iris dataset is a common toy example.
