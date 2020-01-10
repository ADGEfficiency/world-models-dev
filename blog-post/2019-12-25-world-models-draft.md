---
title: 'Reimplementing World Models'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: A TensorFlow 2.0 reimplementation of Ha & Schmidhuber (2018).

---


# The environment

## Working with `car-racing-v0`

I used the same version of OpenAI gym as the paper (`gym==0.9.4`).

In the `car-racing-v0` environment, the agents **observation** is raw image pixels (96, 96, 3) - this is cropped and resized to (64, 64, 3).

<center>
	<img src="/assets/world-models/f1-final.png">
	<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

Of particular importance is using the following in the `reset()` and `step()` methods ([see the GitHub issue here](https://github.com/openai/gym/issues/976)).  If you don't use it you will get corrupt environment observations!

```python
#  do this before env.step(), env.reset()
self.viewer.window.dispatch_events()
```

The **action** has three continuous dimensions - `[steering, gas, break]` (could give min / max).

The **reward** function is 
- -0.1 for each frame
- +1000 / N for each tile visited (N = total tiles on track)

Evolution = only learn from total episode reward

Of particular interest is a hyperparameter controlling the episode length (known more formally as the horizon).  This is set to 1,000 throughout the paper codebase.  Changing this can have interesting effects on agent performance.

Below the code for sampling environment observations is given in full ([see the source here](https://github.com/ADGEfficiency/world-models-dev/blob/master/worldmodels/dataset/car_racing.py)):

```python
# worldmodels/dataset/car_racing.py
```

# The Agent

F1

# Vision

### Why do we need to see?

Why vision is important?  The most common operation in modern computer vision is **dimensionality reduction**.  

Why is dimensionality reduction important?  **Decisions are easier in low dimensional spaces.  Prediction is also easier**!

Let's take our `car-racing-v0` environment .  A low dimensional representation of the environment observation might look something like:

```python
observation = [
	on_road = 1,
	corner_to_the_left = 1,
	corner_to_the_right = 0
]
```

Using these three numbers, we could imagine deriving a control policy.  Try to do this with 27,648 numbers arranged in a shape (96, 96, 3)!

We have a definition of vision as reducing dimensionality.  How does our agent see?

### What the controller sees

The vision of the World Models agent reduces the environment observation $x$ (96, 96, 3) into a low dimensional representation $z$ (32,) known as the **latent space**.

The latent representation is **hidden**.  It is unobserved - we have no labels for these 32 variables.  The controller uses the latent representation $z$ as one of its inputs.  It never uses $x$ (or it's reconstruction $x'$) directly.

How do we learn this latent representation if we don't have examples?  One technique is a Variational Autoencoder.

## The Variational Autoencoder

A **Variational Autoencoder (VAE)** forms the vision of our agent.

The VAE is a **generative** model that learns the data generating process.  The data generating process is $P(x,z)$ - the joint distribution over our data (the probability of $x$ and $z$ occurring together).  

The VAE uses **likelihood maximization** to learn this joint distribution $P(x,z)$.  Likelihood maximization is the process of maximizing the similarity between two distributions.  In our case these distributions are the distribution over our training data (the data generating process) and our parametrized approximation (a convolutional neural network).

Let's start with the definition of a conditional probability:

$$ P(x \mid z) = \frac{P(x, z)}{P(z)} $$

Rearranging this definition gives us a decomposition of the joint distribution  (this is the Product rule):

$$P(x, z) = P(x \mid z) \cdot P(z)$$

This decomposition describes the entire generative process:
- first sample a latent representation

$$z \sim P(z)$$

- then sample a generated data point $x'$, using the conditional probability $P(x \mid z)$

$$x' \sim P(x \mid z)$$

These sampling and decoding steps describe the generation of new data $x'$.  It doesn't describe the entire structure of the VAE (more on that later).

Where to generative models fit in the context of other supervised learning methods?

### Generative versus discriminative models

All approaches in supervised learning can be categorized into either generative or discriminative models.

We have seen that generative models learn a **joint distribution** $P(x, z)$ (the probability of $x$ and $z$ occurring together).  Generative models allow generation, as we can sample from the learnt joint distribution.

Discriminative models learn a **conditional probability** $P(x \mid z)$ (the probability of $x$ given $z$).  Discriminative models are used for prediction, using observed $z$ to predict $x$.  This is simpler than generative modelling.

A common discriminative computer vision problem is classification, where a high dimensional image is fed through convolutions and outputs a predicted class.

Now we understand the context of generative models in supervised learning, we can look at the context of the VAE within generative models.

### The VAE in context

The VAE sits alongside the Generative Adversarial Network (GAN) as the state of the art in generative modelling.  GANs typically outperform VAEs on reconstruction quality, with the VAE providing better support over the data.  By support, we mean the number of different values a variable can take.

<center>
	<img src="/assets/world-models/gan.png">
	<figcaption>
		Progress in GANS - <a href="https://www.iangoodfellow.com/slides/2019-05-07.pdf">Adverserial Machine Learning - Ian Goodfellow - ICLR 2019</a>
	</figcaption>
  <div></div>
</center>

The VAE has less in common with classical (such as sparse or denoising) autoencoders, which both require the use of the computationally expensive Markov Chain Monte Carlo.

### What makes the VAE a good choice for the World Models agent?

A major benefit of generative modelling is the ability to generate new samples $x'$.  Yet our World Models agent never uses $x'$ (whether a reconstruction or a new sample).

The role of the VAE in our agent is to provide a compressed representation $z$ by learning to encode and decode a latent space.

The lower dimensional latent space is easier for our memory and controller to work with.

What qualities do we want in our latent space?  One is **meaningful grouping**.  This requirement is a challenge in traditional autoencoders, which tend to learn spread out latent spaces.

Meaningful grouping means that similar observations exist in the same part of the latent space, with samples that are close together in the latent space producing similar images when decoded.  This grouping means that even observations that the agent hadn't seen before could be responded to the same way.

Meaningful grouping also allows **interpolation** - meaning that we can understand observations we haven't seen before, if they are encoded close to observations we have seen before.

So how do we get meaningful encoding of an observation?  One technique is to constrain the size of the latent space (32 variables for the World Models VAE).  The VAE puts even more effort into TODO

## VAE structure

The VAE is formed of three components - an encoder, a latent space and a decoder.

f2

### Encoder

The primary function of the encoder is **recognition**.  The encoder is responsible for recognizing and encoding the hidden **latent variables**.

The encoder is built from convolutional blocks that map from the input image ($x$) (64, 64, 3) to statistics (means & variances) of the latent variables (length 64 - 2 statistics per latent space variable).

### Latent space

Constraining the size of the latent space (length 32) is one way auto-encoders learn efficient compression of images.  All of the information needed to reconstruct a sample $x$ must exist in only 32 numbers!

The statistics parameterized by the encoder are used to form a distribution over the latent space - a diagonal Gaussian.  A diagonal Gaussian is a muntivariate Gaussian with a diagonal covariance matrix.  This means that each variable is independent.

(is this enforcing a gaussian prior or posterior?)

This parameterized Gaussian is an approximation.  Using it will limit how expressive our latent space is, 

$$z \sim P(z \mid x)$$

$$ z \mid x \approx \mathbf{N} (\mu_{\theta}, \sigma_{\theta}) $$

$$z \sim E(\mathbf{N} (\mu_{\theta}, \sigma_{\theta}))$$

We can sample from this latent space distribution, making the encoding of an image $x$ stochastic.

Because the latent space fed to the decoder is spread (controlled by the parameterized variance of the latent space), it learns to decode a range of variatons for a given $x$.

Ha & Schmidhuber propose that the stochastic encoding leads to a more robust controller in te agent.

### Decoder

The decoder uses deconvolutional blocks to reconstruct the sampled latent space $z$ into $x'$.  In the World Models agent, we don't use the reconstruction $x'$ - we are interested in the lower dimensional latent space representation $z$.

The agent uses the latent space is used in two ways
- directly in the controller
- as features to predict $z'$ in the memory

But we aren't finished with the VAE yet - in fact we have only just started.

## The three forward passes

Now that we have the structure of the VAE mapped out, we can be specific about how we pass data through the model.

### Compression 

$x$ -> $z$

- encode an image $x$ into a distribution over a low dimensional latent space
- sample a latent space $z \sim E_{\theta}(z \mid x)$

### Reconstruction

$x$ -> $z$ -> $x'$

- encode an image $x$ into a distribution over a low dimensional latent space
- sample a latent space $z \sim E_{\theta}(z \mid x)$
- decode the sampled latent space into a reconstructed image $x' \sim D_{\omega}(x' \mid z)$

### Generation

$z$ -> $x'$

- sample a latent space $z \sim P(z)$
- decode the sampled latent space into a reconstructed image $x' \sim D_{\omega}(x' \mid z)$

## The backward pass

We do the backward pass to learn - maximizing the joint likelihood of an image $x$ and the latent space $z$.

Let's start with the encoder. We can write the encoder $E_{\theta}$ as model that given an image $x$, is able to sample the latent space $z$.  The encoder is parameterized by weights $\theta$:

$$ z \sim E_{\theta}(z \mid x) $$

The encoder is an approximation of the true posterior $P(z \mid x)$ (the distribution that generated our data).  Bayes Theorem shows us how to decompose the true posterior:

$$P(z \mid x) = \dfrac{P(x \mid z) \cdot P(z)}{P(x)}$$

The key challenge is calculating the posterior probability of the data $P(x)$ - this requires marginalizing out the latent variables. Evaluating this is exponential time:

$$P(x) = \int P(x \mid z) \cdot P(z) \, dz$$

The VAE sidesteps this expensive computation by *approximating* the true posterior $P(z \mid x)$ using a diagonal Gaussian:

$$ x \mid z \sim \mathbf{N} \Big(\mu_{\theta}, \sigma_{\theta}\Big) $$

$$P(x \mid z) \approx E(x \mid z ; \theta) = \mathbf{N} \Big(x \mid \mu_{theta}, \sigma_{\theta}\Big)$$

This approximation is **varational inference** - using a family of distributions (in this case Gaussian) to approximate the latent variables.  Using variational inference is is a key contribution of the VAE.

Now that we have made a decision about how to approximate the latent space distribution, we want to think about how to bring our parametrized latent space $E_{\theta}(z \mid x)$ closer to the true posterior $P(z \mid x)$.

In order to minimize the difference between our two distributions, we need way to measure the difference.

The VAE uses a Kullback-Leibler divergence ($\mathbf{KLD}$).  The $\mathbf{KLD}$ has a number of interpretations:
- measures the information lost when using one distribution to approximate another
- measures a non-symmetric difference between two distributions
- measures how close distributions are

$$\mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z \mid x) \Big) = \mathbf{E}_{z \sim E_{\theta}} \Big[\log E_{\theta}(z \mid x) \Big] - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log P(x, z) \Big] + \log P(x)$$

This $\mathbf{KLD}$ is something that we can minimize - it is a loss function.  But our exponential time $P(x)$ (in the form of $\log P(x)$) has reappeared!

Now for another trick from the VAE.  We will make use of
- the Evidence Lower Bound ($\mathbf{ELBO}$)
- Jensen's Inequality

The $\mathbf{ELBO}$ is given as the expected difference in log probabilities when we are samlping our latent vectors from our encoder $E_{theta}(z \mid x)$:

$$\mathbf{ELBO}(\theta) = \mathbf{E}_{z \sim E_{\theta}} \Big[\log P(x,z) - \log E_{\theta}(z \mid x) \Big]$$

Combining this with our $\mathbf{KLD}$ we can form the following:

$$\log P(x) = \mathbf{ELBO}(\theta) + \mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z \mid x) \Big) $$

Jensen's Inequality tells us that the $\mathbf{KLD}$ is always greater than or equal to zero. Because $\log P(x)$ is constant (and does not depend on our parameters $\theta$), a large $\mathbf{ELBO}$ requires a small $\mathbf{KLD}$ (and vice versa).

Remember that we have a $\mathbf{KLD}$ we want to minimize!  We have just shown that we can do this by ELBO maximization.  After a bit more mathematical massaging (see the excellent [Altosaar - Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)) we arrive at:

$$ \mathbf{ELBO}(\theta, \omega) = \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\omega}(x' \mid z) \Big] - \mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z) \Big) $$

Note the appearance of our decoder $D_{\omega}(x \mid z)$.  The decoder is used to approximate the true posterior $P(x' \mid z)$ - the conditional probability distribution over the reconstruction of latent variables into a generated $x'$ (given $x$).

The last step is to convert this $\mathbf{ELBO}$ maximization into a more familiar loss function minimization.  We now have the VAE loss function's final mathematical form - in all it's tractable glory:

$$ \mathbf{LOSS}(\theta, \omega) = - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\omega} (x' \mid z) \Big] + \mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big)  $$

The loss function has two terms - the log probability of the reconstruction (aka the decoder) and a $\mathbf{KLD}$ between the latent space (sampled from our encoder) and the latent space prior $P(z)$.

Remember that the loss function above is the result of minimizing the $\mathbf{KLD}$ between our encoder $E_{\theta}(z \mid x)$ and the data generating distribution $P(z \mid x)$.  What we have is a result of maximizing the log-likelihood of the data.

## Implementing the loss function in code

Although our loss function is in it's final mathematical form, we will make three more modifications before we implement it in code:
- convert the log probability of the decoder into a pixel wise reconstruction loss
- use a closed form solution to the $\mathbf{KLD}$ between our encoded latent space distribution and the prior over our latent space $P(x)$
- refactor the randomness using reparameterization

We will look at these two modifications in terms of the two terms in our loss function (??)

### First term - reconstruction loss

$$ - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\omega} (x' \mid z) \Big] $$

The first term in the VAE loss function is the log-likelihood of reconstruction - given latent variables $z$, the distribution over $x'$.  The latent variables are sampled from our encoder (hence the expectation $\mathbf{E}_{z \sim E_{\theta}}$).

Minimizing the negative log-likelihood is equivilant to likelihood maximization.  In our case, the likelihood maximization maximizies the similarity between  the distribution over our training data $P(x \mid z)$ and our parametrized approximation.

Section 5.1 of the fountional [Deep Learning textbook](http://www.deeplearningbook.org/)  that for a Gaussian approximation, maximizing the log-likelihood is equivilant to minimizing mean square error ($\mathbf{MSE}$):

$$\mathbf{MSE} = \frac{1}{n} \sum \Big( \mid \mid x' - x \mid \mid \Big)^{2} $$

(Remember we sample our latent variables from our encoder, which is a Gaussian approximation).

In practice this term is often implemented in code as a **pixel wise reconstruction loss** (also known as an L2 loss):

```python

```

High quality reconstructions mean that the VAE has learnt an efficient encoding of a given sample image $x$.

### Second term - regularization

$$ \mathbf{LOSS}(\theta, \omega) = - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\omega} (x' \mid z) \Big] + \mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big)  $$

The intuition of the second term in the VAE loss function is compression or regularization.

The second term in the VAE loss function is the $\mathbf{KLD}$ between the and the latent space prior $P(z)$.  

We haven't yet specified what the prior over the latent space should be.  A convenient choice is a **Standard Normal** - a Gaussion with a mean of zero, variance of one.

Minimizing the $\mathbf{KLD}$ means we are trying to make the latent space look like random noise.  It encourages putting encodings near the center of the latent space.

The KL loss term further compresses the latent space.  This compression means that using a VAE to generate new images requires only sampling from noise!  This ability to sample without input is the definition of a generative model.

Because we are using Gaussians for the encoder $E_{theta}(z \mid x)$ and the latent space prior $ P(z) = \mathbf{N} (0, 1) $, the $\mathbf{KLD}$ has closed form solution ([see Odaibo - Tutorial on the VAE Loss Function](https://arxiv.org/pdf/1907.08956.pdf)).

$$\mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big) = \frac{1}{2} \Big( 1 + \log(\sigma_{\theta}^{2}) - \sigma_{\theta}^{2} - \mu_{\theta} \Big)$$

This is how the $\mathbf{KLD}$ is implemented in the VAE loss:


```python

```

A note on the use of $\log \sigma^{2}$ - we force our network to learn this by taking the exponential later on in the program:

```python

```

### Reparameterization trick

Because our encoder is stochastic, we need one last trick - a rearrangement of the model architecture, so that we can backprop through it.  This is the **reparameterization trick**, and results in a latent space architecture as follows:

fig - 2.3 2019, 4 in 2016

$$ n \sim \mathcal{N}(0, 1) $$

$$ z = \sigma_{theta} (x) \cdot n + \mu_{theta} (x) $$

After the refactor of the randomness, we can now take a gradient of our loss function and train the VAE.

## Vision - Summary

The contributions of the VAE are:
- variational inference to approximate
- compression / regularization of the latent space using a KLD between our learnt latent space and a prior $P(z) = \mathbf{N} (0, 1)$
- stochastic encoding of a sample $x$ into the latent space $z$ and into a reconstruction $x'$

The reasons why we use the VAE in the World Models agent:
- learn the latent representation $z$

```python
# worldmodels/vision/vae.py

# worldmodels/vision/train_vae.py
```

# Memory

Conditional, discriminative

$$ P(z' | z, a) $$

The primary role of the memory in the World Models agent is **prediction**.  The reason for this is that prediction is useful for control.

This is one of the most important lessons for a data scientist - so important that it defines data science for me

>> DEFN

Compression of time (use lstm hidden state) - predict next step, but learn a longer representation of time via hidden state (specifically $h$)

Fig

##  Mixtures

Arbitrary conditional probability distributions
- Flex to model general distribution functions
- no assumption of independent variables in the pred
- use variance as measure of uncertainty

A primary motivation behind using a mixture of distributions is that we can approximate **multi-modal** distributions.

Probability distribution output by a mixture can (in principle!) be calculated.  The flexibility is similar to a feed forward neural network, and likely has the same distinction between being able approximate versus being able to learn.


Bishop (?) shows that by training a neural network using a least squares loss function, we are able to learn two statistics.  One is the conditional mean, which is our prediction.  The second statistic is the variance, which we can approximate from the residual.

We can use these two statistics to form a Gaussian.

Our kernel of choice is the Gaussian, which has a probability density function:

$$ \phi (z' \mid z, a) = \frac{1}{\sqrt(2 \pi) \sigma(z, a)} \exp \Bigg[ - \frac{\lVert z' - \mu(z, a) \rVert^{2}}{2 \sigma(z, a)^{2}} \Bigg] $$

Prediction = linear combination of Gaussians
- means
- variance
- mix probs (need a softmax, exponential (param = log sigma :), become $\pi$ via bayes)

Exponential of the sigmas

The parametrized mixture probabilities $\pi{\theta}$ are priors of the target having been generated by a mixture component.  These are transformed via a softmax:

$$ \pi_{\theta} = \frac {\exp (z)}{\sum_{mixes} exp(z)} $$

This then means our mixture satasfies the constraint:

$$ \sum_{mixes} \pi(z, a) = 1 $$

As with the VAE, the memory parameters are found using likelihood maximization.  One interpretation of likelihood maximization is reducing dissimilarity (Goodfellow)
The parameters $\theta$ are found using likelihood maximization.  

$$ M(z' \mid z, a) =  \sum_{mixes} \alpha(z, a) \cdot \phi (z'| z, a) $$

$$ \mathbf{LOSS} = - \log M(z' \mid z, a)$$

$$ \mathbf{LOSS} = - \log  \sum_{mixes} \alpha(z, a) \cdot \phi (z'| z, a) $$

Sum or select (I select)

How do we set these statistics? answer = lstm

```python
#worldmodels/memory/memory.py - GaussianMix
```

## LSTM

Motivation behind using an LSTM
- non-linear
- models sequential structure

Use NN -> non-linear

$$ M_{\theta}(z'| z, a, h) $$

cell state, h state

```python
#worldmodels/memory/memory.py - LSTM
```

## Putting the Memory together

Testing with notebooks.
