# Mini-Batch Gradient Descent
 At each step #mini-batch-gradient-descent computes the gradient on a **small random set of instances** called a *mini-batch*.

Concretely, it computes the gradient over a mini-batch $B$ of $n_b$ instances:
$$
\nabla_{\boldsymbol{\theta}}\,\text{MSE}_B(\boldsymbol{\theta}) = \frac{2}{n_b} \mathbf{X}_B^\intercal \left( \mathbf{X}_B \boldsymbol{\theta} - \mathbf{y}_B \right)
$$
This sits between the two extremes: $n_b = m$ recovers [[Batch GD|batch gradient descent]], $n_b = 1$ recovers [[Stochastic GD|SGD]].
 
 Its main advantage over #SGD is a **performance boost from hardware acceleration** of matrix operations, especially on GPUs. Also, its progress is less erratic than SGD (more so with larger mini-batches), so it ends up walking a bit closer to the minimum — but it may also find it harder to escape local minima on problems that have them.

The next figure shows the paths of all three variants in parameter space: batch GD's path stops at the minimum, while SGD and mini-batch GD keep walking around it (though a good learning schedule would let them settle too).
![[4-11.png]]