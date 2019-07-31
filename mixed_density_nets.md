Bishop 1994

Data generation = density of $p(x,t)$

Often convenient to decompose the joint prob density into product of conditional density of the target, and unconditioned input

$p(x,t) = p(x|t) \cdot p(x)$

For the purpose of making predictions of t for new values of x, we want $p(t|x)$

$$loss = /frac{1}{2} /sum (prediction - target)^2$$

Mixed Density = framework for modelling conditional density functions

Figure 1 is quite interesting - TODO

For classification, cross entropy loss leads to conditional averages of the posterior probabilities (which is optimal)

Posterior probability = conditional probability dependent on evidence / background
- non-posterior prob of person finding treasure if digging in random spot
- posterior prob if they dig in spot where metal detector rings

 For regression, usually minimizing sum of squares
- network learns to approximate the conditional mean of the target (conditioned on the input $x$)
- this average is often a limited description of the target
- especially for distributions with multiple modes (multivalued)

Perfect MSE optimization learns two statistics with MSE - the conditional average and the average variance (which is our residual error at the minimum of the MSE loss)
- knowing these two stats, we can model a Gaussian
- but normal nn makes no assumption about the distribution

Mixed Density Nets makes an assumption that the conditional distribution is Gaussian
- use maximum likelihood to obtain the least squares formalism

Assume taht the target data is governed by 

$$p(t|x) = /frac{1}{(2 \pi)^{1/2} \sigma} exp(- \frac{[F(x) - t]^2}{2 \sigma^2})$$

` probability = (1 / (2 * pi)^2 sigma) exp(-(mean-target)^2 / 2 * sigma) `

$F(x)$ is the mean of the target variable - a function of x
- model using a function (ie neural network)

Parameters for this function determined by maximizing the likelihod that the model gave rise to the data points
- likelihood of dataset = product of the likelihoods for each datapoint
- minimize the negative log of this = our error function

Which then leads to least squares
- proving that we can derive least squares from maximum likelihood of an assumed Gaussian

But we want to do better than least squares -> motivation for mixed density

Mixture model = linear combination of kernels

$p(t|x) = \sum \alpha(x) \phi(t|x)$

alpha = mixture coefficient




