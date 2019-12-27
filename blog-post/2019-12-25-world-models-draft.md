---
title: 'World Models (2018) Reimplementation'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: Taking the advice of Open AI and building  a high quality implementation in TensorFlow 2.

---

## Vision

A Variational Autoencoder (VAE) forms the vision of the World Models agent.  

Before we detail the specific contributions of the VAE, it is worth reflecting on what properties we want from a generative model in general:
- high quality reconstructions
- ability to sample new images
- continuous and dense latent space
- meaningful interpolation

When it comes to the World Models VAE, we are particularly interested in the quality of the latent space.


In the `car-racing-v0` environment, the agents observation is raw image pixels.

The `car-racing-v0` environment requires an agent that can see.

Learning from high dimensional data is challenging, but it is well within the wheelhouse of modern machine learning.  2013 Deep Mind.

The VAE is used to provide a compressed representation of the environment observation $x$ as a latent space $z$.

The VAE is a generative model that learns the data generating process.

Learning the joint distribution is allows generative models to generate.

The data generating process is the joint distribution over all variables $P(x, y)$ (the probability of $x$ and $y$ occuring together).  This can be directly contrasted with disciminative models, which learn the simpler conditional probability $P(y \mid x)$ (the probability of $y$ given $x$).

Any vision == dim reduction

A common disriminative computer vision problem is classification, where a high dimensional image is fed through convolutions and outputs a class prediction.  The output exist in a lower dimensional space.

World Models uses dimensionality reduction in the same way - to reduce high dimensional data into a lower dimensional space that is easier to take actions in.

The VAE is used to provide a compressed representation of the environment observation $x$ as a latent space $z$.

The VAE is a likelihood maximization model - maximizing the joint probability of an image $x$ and a latent state $z$:

$$P(x,z)$$

## VAE structure

The VAE is formed of three components - an encoder, a latent space and a decoder.

### Encoder

The primary function of the encoder is **recognition**.  The encoder is responsible for recognizing and encoding hidden **latent variables**.

For a given sample $x$ we don't know the values of the latent variables.  We don't even need to confirm that they exist!

In the `Car-Racing-v0` environment, latent variables could be if the car is off the track, or if the track is bending.  

The encoder is built from convolutional blocks that map from the input image ($x$) (60 x 80 x 3) to statistics (means & variances) of the latent variables (length 64 - 2 per latent space dimension).

### Latent space

Constraining the size of the latent space (length 32) is one way auto-encoders are forced to learn an efficient compression of images.  All of the information needed to reconstruct a sample $x$ must exist in only numbers!

The statistics parameterized by the encoder are used to form a distribution over the latent space - formed from indepedent Gaussians.  Enforcing a Gaussian prior over the latent space will limit how expressive our latent space distribution is, but improves it's robustness.

$$z \sim P(z \mid x)$$

$$z \sim P(N(\mu, \sigma))$$



### Decoder

We can sample from this latent space distribution.  Once a latent space has been sampled, the decoder uses deconvolutional blocks to reconstruct the original image $x$ into $x'$.

In the World Models agent, we don't use the reconstruction for control - we are interested in the lower dimensional latent space representation $z$.  It is more useful for control.

The reason that we can use noise to generate images is that we pass that noise through the complex function that is the decoder.

## The forward passes

### Forward pass - reconstruction

- encode an image into a distribution over a low dimensional latent space
- sample a latent space $z \sim p(z)$
- decode the sampled latent space into a reconstructed image $x \sim p(x \mid z)$

### Forward pass - generation

- sample a latent space $z \sim p(z)$
- decode the sampled latent space into a reconstructed image $x \sim p(x \mid z)$

We can rewrite the joint probability of our VAE as (check)

$p(x,y) = p(x \mid z)p(z)$

$z \sim p(z)$

$x' \sim p(x \mid z)$

Note that this describes only the latent space and encoder - we don't need the encoder to generate new images (only to train).


## The backward pass

We do forward passes to compress or to generate.  

We do the backward pass to learn - maximizing the likelihood (joint prob) of an image $x$ and the latent space $z$.

Let's start with the encoder. We can write the encoder $E$ as model that given an image $x$, is able infer the statistics of the latent space $z$:

$$ E(z \mid x) $$

$$ E(N(\mu_{z}, \sigma_{z}) \mid x) $$

Bayes Theorem shows us how to decompose the encoder:

$$p(z \mid x) = \dfrac{p(x \mid z) \cdot p(z)}{p(x)}$$

The key challenge is calculating the probability of the data $p(x)$ - this requires evaluating an exponential time integral.  

The VAE sidesteps this by approximating the true posterior $p(z \mid x)$ using Gaussians.  

This is the **varational inference** part of the VAE.  Now that we have made a decision about how to approximate the latent space distribution, we want to think about how to bring our contstrained latent space closer to the true posterior $p(z \mid x)$.

In order to minimize the difference between our tow distributions, we need a distance measure of how our encoder approximates the true posterior $p(z 
/mid x)$.

We can measure this using the Kullback-Leibler divergence (KLD).

$ KLD (q(z \mid x) \mid \mid e(z \mid x)) = E_p[log e(z \mid x)] - E_p[log p(x, z)] + log p(x)$

The KLD has a number of interpretations:
- measures the information lost when using q to approximate p
- measures a non-symmetric difference between two distributions

We now have a loss function - a difference between our parameterized latent space distribution $E(z \mid x)$ and the true distribution $p(z \mid x)$.

Now for another trick from the VAE, which results in replacing computing and minimizing this KLD with Evidence Lower Bound (ELBO) maximization, .  Expanding the notation to show the parameters of the encoder ($\theta$) and the decoder ($\omega$):

$ELBO(\theta, \omega) =  E_{z \sim e_{\theta}(z \mid x)} [\log d(x \mid z)] $

The last step is to convert the ELBO maximization into the more familiar loss function minimization, which results in the VAE loss function's final form:

$ L(\theta, \omega) = - above $

Remember that the loss function above is the result of minimizing the KLD between our encoder $e(z \mid x)$ and the true distribution $p(z \mid x)$.  What we have is a result of maximizing the log-likelihood of the data.


### First term

log prob == pixel wise

High quality reconstructions mean that the VAE has learnt an efficient encoding of a given sample image $x$.  

When we are training the encoder, the goal is to produce a model that learns $P(z \mid x)$ - a model that is able to infer good values of the latent variables given our observed data.







Pixel wise loss stuff

https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencode://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder

This can be done because

$x \mid z \sim N(u, sig)$

which is normal -> log likelihood of gaussian distribution and L2 loss

https://stats.stackexchange.com/questions/288451/why-is-mean-squared-error-the-cross-entropy-between-the-empirical-distribution-a/288453

Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model. For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model.
In practice this term is implemented using a pixel wise reconstruction loss.  https://stats.stackexchange.com/questions/288451/why-is-mean-squared-error-the-cross-entropy-between-the-empirical-distribution-a/288453

This leads to the first term in the VAE loss function - the negative log likelihood of the decoder, when sampling from a latent space parameterized by the encoder:

### Second term

Plot of my images in 2D latent space tsne (do by hand)

The most useful interpretation of the second term in the VAE loss function is compression.

Generating new samples requires a latent space that is continuous, with samples that are close together in the latent space producing similar images when decoded.  This requirement is a challenge in traditional autoencoders, which learn spread out latent spaces.

Compression of the latent space allows interpolation.

The VAE tackles this problem by making the encoding stochastic.  Because the latent space fed to the decoder is spread (controlled by the parameterized variance of the latent space distribution), it learns to decode a range of variatons for a given $x$.

Thus the latent space being stochastic helps to make it continuous.  This latent space also requires compression.  Traditional autoencoders compress the latent space by constraining it to a fixed length.  The VAE takes it one step further by including second term in the loss function - a Kulback-Lieber divergence (KLD).

The KLD is a measurement of how different two probability distributions are (note I didn't say distance!).  But which two?

The first term is the latent space - intutive as we want to improve the quality of the latent space.

The second term in the VAE KLD is the standard normal distribution (a normal with mean of zero, variance of one).  

Minimizing the KLD means we are trying to make the latent space look like random noise.  It encourages putting encodings near the center of the latent space. 

The KL loss term further compresses the latent space.  This compression means that using a VAE to generate new images requires only sampling from noise!  This ability to sample without input is the definition of a generative model.

## Stochastic encoding

Interperolation further helped by 

VAE being stochastic == more robust controller

### Reparameterization trick

reparameterization (allows stochastic gradients)
- 2.3 2019, 4 in 2016

The VAE is therefore stochastic - the latent space is sampled from a distribution parameterized by the encoder.  This sampling requires a reorganization of the latent space from within the model internals to an input to the model.

The reparameterization trick results in a latent space architecture as follows:

$ z = \sigma (x) \cdot n + \mu (x) $

$ n \sim \mathcal{N}(0, 1) $

After the refactor of the randomness, we can now take a gradient of our loss function and train the VAE.  Remember how the VAE integrates into the larger World Models agent - we never use the reconstruction - we only want the latent space.





