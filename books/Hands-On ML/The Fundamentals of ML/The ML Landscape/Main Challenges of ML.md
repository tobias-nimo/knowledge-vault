# Main Challenges of Machine Learning
In short, since your main task is to select a model and train it on some data, the two things that can go wrong are **“bad model”** and **“bad data”**.
## Insufficient Quantity of Data
It takes a lot of data for most machine learning algorithms to work properly. Even for very simple problems you typically need **thousands of examples**, and for complex problems you may need **millions of examples**.

> The 2009 paper [[The Unreasonable Effectiveness of Data]] established the idea that data matters more than algorithms for complex problems.
## Non-representative Data
It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have **sampling noise** (i.e., non-representative data as a result of chance), but even very large samples can be non-representative if the sampling method is flawed. This is called [[Sampling Bias]].
## Poor-Quality Data
Obviously, if your training data is full of **errors**, **outliers**, **missing values** and **noise**, it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well. It is often well worth the effort to spend time **cleaning up your training data**.
## Irrelevant Features
A machine learning system can only learn effectively if it is trained on a **set of relevant features** with few or no irrelevant ones. Ensuring high-quality features is the goal of #feature-engineering:
- Feature selection
- Feature extraction
- Creating new features by gathering new data
## Overfitting
#overfitting means that the model **performs well on the training data, but it does not generalize well**.
![[1-23.png]]
**Complex models** can detect subtle patterns in the data, but if the training set is noisy, or if it is too small then the model is likely to detect patterns in the noise itself. Obviously these patterns will not generalize to new instances.

Here are possible solutions:
- Simplify the model
	- by selecting one with fewer parameters,
	- by reducing the number of attributes in the training data, or
	- by constraining the model.
- Gather more training data.
- Reduce the noise in the training data (e.g., fix data errors and remove outliers).
## Underfitting
#underfitting is the opposite of overfitting: it occurs when your **model is too simple to learn the underlying structure of the data**. 
![[G-1.png]]
Here are the main solutions:
- Select a more powerful model, with more parameters.
- Feed better features to the learning algorithm (feature engineering).
- Reduce the constraints on the model (for example by reducing the regularization
hyperparameter).
## Deployment Issues
Even if you have a large and clean dataset and you manage to train a beautiful model that neither underfits nor overfits the data, you may still run into issues during deployment: for example, the model may be too complex to maintain, or too large to fit in memory, or too slow, or it may not scale properly, it may have security vulnerabilities, it may become outdated if you don’t update it often enough, etc.

To summarize, a machine learning system will not perform well if the training set is too small, unrepresentative, noisy, or polluted with irrelevant features—garbage in, garbage out. The model must also strike the right balance between simplicity and complexity: if it is too simple, it will underfit; if it is too complex, it will overfit. Finally, deployment constraints should be carefully considered when designing the solution.