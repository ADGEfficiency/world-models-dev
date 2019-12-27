---
title: 'World Models'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: Re.

---

## Vision

In the `car-racing-v0` environment, the agents observation is raw image pixels.  Learning from pixels is challenging, as an image is high dimensional (raw == ?).  The `car-racing-v0` environment requires an agent that can see.

A Variational Autoencoder (VAE) forms the vision of the World Models agent.  

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

Constraining the size of the latent space (length 32) is one way auto-encoders learn efficient compressions.  

The statistics parameterized by the encoder are used to form a distribution over the latent space - formed from indepedent Gaussians.  Enforcing a Gaussian prior over the latent space will limit how expressive our latent space distribution is, but improves it's robustness.

$$z \sim P(z \mid x)$$

$$z \sim P(N(\mu, \sigma))$$

### Decoder

We can sample from this latent space distribution.  Once a latent space has been sampled, the decoder uses deconvolutional blocks to reconstruct the original image $x$ into $x'$.

## The forward passes

### Forward pass - reconstruction

- encode an image into a distribution over a low dimensional latent space
- sample a latent space $z \sim p(z)$
- decode the sampled latent space into a reconstructed image $x \sim p(x \mid z)$

### Forward pass - generation

We can rewrite the joint probability of our VAE as (check)

$p(x,y) = p(x \mid z)p(z)$

$z \sim p(z)$

$x' \sim p(x \mid z)$

Note that this describes only the latent space and encoder - we don't need the encoder to generate new images (only to train).
- sample a latent space $z \sim p(z)$
- decode the sampled latent space into a reconstructed image $x \sim p(x \mid z)$


## The backward pass

We do forward passes to compress or to generate.  

We do the backward pass to learn - maximizing the likelihood (joint prob) of an image $x$ and the latent space $z$.

Let's start with the encoder. We can write the encoder as model that given an image $x$, is able infer the statistics of the latent space $z$:

$$ E(z /mid x) $$

$$ E(/mu_{z}, /sigma_{z}, /mid x) $$

Bayes Theorem shows us how to decompose the encoder:

$p(z \mid x) = \dfrac{p(x \mid z) \cdot p(z)}{p(x)}$

The key challenge is calculating the probability of the data $p(x)$ - this requires evaluating an exponential time integral.  

The VAE sidesteps this by approximating the true posterior $p(z \mid x)$ using Gaussians.

How well does our encoder approximate the true posterior $p(z|x)$?  We can measure this using the Kullback-Leibler divergence (KLD).

$ KLD (q(z \mid x) \mid \mid e(z \mid x)) $

$= E_p[log e(z \mid x)] - E_p[log p(x, z)] + log p(x)$

The KLD has a number of interpretations:
- measures the information lost when using q to approximate p
- measures a non-symmetric difference between two distributions

elbo


