http://gregorygundersen.com/blog/2018/04/29/reparameterization/

https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73 (bit on reparameterization)

[Intuitively Understanding Variational Autoencoders - Irhum Shafkat](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

papers!

### Generative models

The VAE is a generative model, which learns a data generating process. Generative models can be contrasted with the more common disciminative models, which map from features to target.

Generative models are the joint distribution over all variables $P(x, y)$ (the probability of $y$ and $x$) rather than the simpler conditional probability $P(y|x)$ (the probability of $y$ given $x$) learnt in by a disciminative model.

Learning the joint distribution is what allows generative models to generate.  The model simulates the data generating process.

The VAE can be compared with the Generative Adverserial Network (GAN).  GANs typically outperform VAEs on reconstruction quality, with a VAE providing better support over the data.

f1 https://www.iangoodfellow.com/slides/2019-05-07.pdf

The VAE has less in common with classical autoencoders (such as sparse or denoising autoencoders), which both require the use of the computationally expensive Markov Chain Monte Carlo.

### Likelihood (loss)

VAE = likelihood based (max likelihood)
- equivilant to minimizing KLD between data & model
- likelihood max = accurate reproduce
 
Evidence lower bound
- parameters optimized on ELBO (lower bounid of marginal likelihood)
fig 2.1, 

gaussians allow optimization, to make training data more likely

need to be able to compute P(X'|z)
