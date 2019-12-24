http://gregorygundersen.com/blog/2018/04/29/reparameterization/

https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73 (bit on reparameterization)

[Intuitively Understanding Variational Autoencoders - Irhum Shafkat](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

papers!

### Generative models


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
