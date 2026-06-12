# Instance-Based vs. Model-Based Learning
Machine Learning systems can be classified by **how they generalize**. There are two main approaches to generalization...
## Instance-based learning
The system **learns the examples by heart**, then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them). A common example is k-nearest neighbors (k-NN).
![[1-15.png]]
Instance-based learning often **shines with small datasets**, especially if the data keeps changing, but it does not scale very well: it requires deploying a whole copy of the training set to production; making predictions requires searching for similar instances, which can be quite slow; and it doesn’t work well with high-dimensional data such as images.
## Model-based learning
The system learns a model from the training data by **identifying patterns and relationships in the data**, then uses the learned model to make predictions on new instances. Common examples are linear regression and Support Vector Machines (SVMs).
![[1-16.png]]
