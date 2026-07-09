# Likelihood
"Probability" and "likelihood" are often used interchangeably in everyday language, but they have very different meanings in statistics. Given a statistical model with parameters $\theta$:

- **Probability** describes how plausible a future outcome $x$ is, knowing the parameter values $\theta$.
- **Likelihood** describes how plausible a particular set of parameter values $\theta$ is, after the outcome $x$ is known.

The likelihood function uses exactly the **same formula** as the probability density function, but treats the variables differently: the #PDF is viewed as a function of $x$ with $\theta$ fixed, whereas the #likelihood is viewed as a function of $\theta$ with $x$ fixed.

For a model with density
$$f(x;θ)$$
the corresponding likelihood function after observing a value $x$ is
$$  
\mathcal{L}(\theta\mid x)=f(x;\theta)
$$
For a dataset of independent observations $\mathbf{X}={x_1,\ldots,x_n}$, the likelihood is the product of the individual probability densities:
$$  
\mathcal{L}(\theta\mid\mathbf{X})  
=\prod_{i=1}^{n}f(x_i;\theta).  
$$

> [!warning]  
> The likelihood function is **not a probability distribution**. A probability density integrates to 1 over all possible values of $x$, whereas the likelihood is a function of $\theta$, and its integral over the parameter space has no special meaning—it can be any positive value.

## Maximum Likelihood Estimation
Given a dataset $\mathbf{X}$, a common task is to estimate the parameter values that **maximize the likelihood function**. This is called the maximum likelihood estimate ( #MLE ):
$$  
\hat{\theta}
=
\arg\max_\theta  
\mathcal{L}(\theta\mid\mathbf{X}).  
$$
In practice, we almost always maximize the **log-likelihood** instead, since it is mathematically equivalent and much easier to compute.
$$  
\log\mathcal{L}(\theta\mid\mathbf{X})  
=\sum_{i=1}^{n}\log f(x_i;\theta).  
$$
Once the estimate $\hat{\theta}$ is found, the maximized likelihood provides a **measure of how well the model fits the observed data**.