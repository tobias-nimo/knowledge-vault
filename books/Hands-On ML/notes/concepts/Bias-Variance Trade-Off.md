# Bias/Variance Trade-Off
An important theoretical result of statistics is that a model's **generalization error** can be expressed as the sum of three very different errors:

- **Bias** — error due to **wrong assumptions**, such as assuming the data is linear when it is actually quadratic. A high-bias model is most likely to #underfit the training data.
- **Variance** — error due to the model's **excessive sensitivity to small variations in the training data**. A model with many degrees of freedom (such as a high-degree polynomia model) is likely to have high variance and thus #overfit the training data.
- **Irreducible error** — error due to the **noisiness of the data itself**. The only way to reduce it is to clean up the data (e.g. fix broken sensors, or detect and remove outliers).

Increasing a model’s complexity will typically increase its variance and reduce its bias.
Conversely, reducing a model’s complexity (or increasing regularization) increases its
bias and reduces its variance. This is why it's called the #bias-variance-trade-off.

![[4-17.png]]
