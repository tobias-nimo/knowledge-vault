# Gradient Descent
Training a model is a search for the parameter combination that minimizes the cost over the training set — **a search in the model's parameter space**. The more parameters, the more dimensions, and the harder the search.

#gradient-descent is a **generic optimization algorithm** that tweaks parameters iteratively to minimize a cost function.

The intuition: imagine you're lost in the mountains in dense fog and can only feel the slope under your feet. To reach the valley quickly, you head downhill in the direction of the steepest slope. Gradient descent does exactly this — it measures the local gradient of the cost function with regard to the parameter vector $\boldsymbol{\theta}$ and moves in the direction of descending gradient. Once the gradient is zero, you've reached a minimum.

In practice you start by filling $\boldsymbol{\theta}$ with random values (*random initialization*), then improve it one baby step at a time, each step trying to decrease the cost (e.g. the [[Linear Regression|MSE]]), until the algorithm converges.
![[4-3.png]]

> [!note] About #MSE 
> Fortunately, the **MSE cost function for [[Linear Regression|linear regression]] is convex**: any line segment joining two points on the curve never dips below it, so there are no local minima — just one global minimum. It's also continuous with a slope that never changes abruptly. Together these mean gradient descent is **guaranteed to approach the global minimum arbitrarily closely**, given enough time and a learning rate that isn't too high.
## Algorithm Variants
GD variants differ in **how much data they use to compute the gradient at each step**, trading off speed per step, regularity of convergence, and the ability to escape local minima:

- **[[Batch GD]]** — uses the **whole training set** every step. Direct and stable, but slow on large sets.
- **[[Stochastic GD]]** — uses **one random instance** per step. Fast and memory-light (out-of-core), bounces around the minimum, and can escape local minima with a good learning schedule.
- **[[Mini-Batch GD]]** — uses **small random batches**. A middle ground that benefits from GPU-accelerated matrix operations.
## Common Pitfalls
### Learning Rate
The step size is proportional to the slope and controlled by the #learning-rate hyperparameter $\eta$, so it gradually shrinks as the cost approaches the minimum.

![[4-4.png]]
> **Learning rate is too small** → many iterations are needed to converge, which takes a long time.

![[4-5.png]]
> **Learning rate is too high** → the algorithm can **diverge**, with larger and larger values, failing to find a good solution.

The choice of learning rate has a major impact on how gradient descent behaves. The figure below shows the first 20 steps of the algorithm when training a linear regression model with three different learning rates:
![[4-8.png]]
> too low (left, slow), good (middle, converges in a few epochs), and too high (right, diverges and jumps all over the place).

> [!tip]
> To find a good #learning-rate, use #grid-search — but cap the number of epochs so it can eliminate models that converge too slowly. 
### Local Minima and Plateaus
Not all cost functions are nice regular bowls like MSE— there may be holes, ridges, and plateaus that make convergence hard. For example, see Figure 4-6:
- If random initialization starts you on the left, you converge to a **local minimum**, which is worse than the global minimum.
- If it starts you on the right, crossing the **plateau** takes a very long time, and if you stop too early you never reach the global minimum.

![[4-6.png]]
### Feature scaling
Even a convex bowl like the MSE can be elongated if the **features have very different scales**. 
![[4-7.png]]
> With features on the same scale (left) gradient descent heads straight for the minimum; with very different scales (right) it first moves almost orthogonally to the goal, then crawls down a near-flat valley — much slower.


>[!warning] Scale your features
> #normalization #standardization 
> 
>When using gradient descent, ensure all features have a similar scale (e.g. with Scikit-Learn's `StandardScaler`), or it will take much longer to converge.
