# Look at the Big Picture

## Frame the Problem
The first question to ask your boss is what exactly the business objective is: **How does the company expect to use and benefit from this model?** Knowing the objective is important because it will determine which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.

The next question to ask your boss is **what the current solution looks like (if any)**. The current situation will often give you a reference for performance, as well as insights on how to solve the problem.

With a clear understanding of the objective and the current solution, you can begin **designing the system**:
1. First, determine what kind of **training supervision** the model will need: is it a supervised, unsupervised, semi-supervised, self-supervised, or reinforcement learning task? 
2. What type of machine learning tasks is it? Is it classification, regression, or something else?
3. Should you use **batch or online learning** techniques?

## Select a Performance Measure
Your next step is to select a performance measure. 

For example, a typical performance measure for regression problems is the **root mean squared error** ( #RMSE ).
$$
\mathrm{RMSE}(X,y,h)=\sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(h(x^{(i)})-y^{(i)}\right)^2}
$$
- $m$ is the number of instances in the dataset.
- $x^{(i)}$ is a vector of all the feature values (excluding the label) of the $i^{(th)}$ instance in the dataset, and $y^{(i)}$ is its label (the desired output value for that instance). 
- $y$ is a vector containing the labels of all the instances in the dataset.
- $X$ is a matrix containing all the feature values (excluding labels) of all instances in the dataset. There is one row per instance, and the $i^{(th)}$ row is equal to the transpose of  $x^{(i)}$, denoted  ${(x^{(i)})}^T$. $$\mathbf{X}=\begin{pmatrix}(\mathbf{x}^{(1)})^{T} \\(\mathbf{x}^{(2)})^{T} \\\vdots \\(\mathbf{x}^{(m)})^{T}\end{pmatrix}$$
- $h$ is your system’s prediction function, also called a hypothesis. When your system is given an instance’s feature vector $x^{(i)}$, it outputs a predicted value $\hat{y}^{(i)}=h(x^{(i)})$ for that instance.
- $\mathrm{RMSE}(X,y,h)$ is the cost function measured on the set of examples using your hypothesis $h$.

If your dataset contains many outliers, you may want to use a different performance metric, since **RMSE is highly sensitive to outliers**. A common alternative is the **Mean Absolute Error (MAE)**, also known as the average absolute deviation.
$$
\mathrm{MAE}(\mathbf{X},y,h)=\frac{1}{m}\sum_{i=1}^{m}\left|h\!\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right|
$$
Both the RMSE and the MAE are [[Ways to Measure Distance]] between two vectors: the vector of predictions and the vector of target values.

## Check the Assumptions
Lastly, it is good practice to list and verify the assumptions that have been made so far (by you or others); this can help you catch serious issues early on.