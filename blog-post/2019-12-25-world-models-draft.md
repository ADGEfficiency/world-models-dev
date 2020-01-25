---
title: 'World Models'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: Ha & Schmidhuber's World Models (2018) reimplemented in TensorFlow 2.0 

---

<center>
	<img src="/assets/world-models/f0.gif">
<figcaption>Performance of the final agent on a conveniently selected random seed. The cumulative episode reward is shown in the lower right.  This agent & seed achieves 893. 900 is solved.</figcaption>
</center>

# Table of Contents

Tense = I, past

- Key Resources
- Motivations
  - why reimplement a paper?
  - why reimplement World Models?
- The Agent
- The Environment
  - `car-racing-v0` as a Markov Decision Process
  - Working with `car-racing-v0`

# Key Resources

- reimplementation code base - [ADGEfficiency/world-models](https://github.com/ADGEfficiency/world-models)
- original paper code base - [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)
- [interactive blog post](https://worldmodels.github.io/) - World Models
- [2018 paper](https://arxiv.org/pdf/1809.01999.pdf) - Recurrent World Models Facilitate Policy Evolution 
- [2018 paper](https://arxiv.org/pdf/1803.10122.pdf) - World Models 
- extensive collection of resources - [ADGEfficiency/rl-resources/world-models](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models)

# Motivations

My main side project in from April 2019 to ??? 2020 was a reimplementation of the 2018 paper by Ha & Schmidhuber.  This project dominated any spare time I had - usually one to three days per month.

<center>
	<img src="/assets/world-models/commits-month.png">
	<figcaption>Commits per month.  Not all commits are made equal.</figcaption>
  <div></div>
</center>

## Why reimplement a paper?

The original idea for reimplementing a paper came from reading an Open AI job advertisement.  Seeing a tangible goal that could put me in the ballpark, I set out looking for a paper to reimplement.

I specifically remember reading *high quality implementation*.  This requirement was echoed in my mind as I developed this project.

## Why reimplement World Models?

There are three papers that have blown me away in the three years I have been working with machine learning.  World Models was the third.

### DQN

The first blew me away without me even reading it.  I have a memory of seeing a YouTube video DQN playing the Atari game Breakout.  Even though I knew nothing of reinforcement learning, the significance of a machine could learn to play a video game from pixels was made clear.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/V1eYniJ0Rnk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
</center>

I had no way of knowing that the algorithm I was watching would be one I implement in four distinct implementations, and teach the ins & outs of DQN and it's evolution into Rainbow.

### AlphaGo Zero

The second was AlphaGo Zero.  The publication of AlphaGo Zero in October 2017 came out after I had taught reinforcement learning twice.

<center>
	<img src="/assets/world-models/Zero_act_learn.png">
	<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

Even at this stage, I didn't fully grasp all of the mechanics in AlphaGo.  But I knew enough to understand the significance of the changes - *tabula rasa* learning among the most important.

### World Models

TODO A PICTURE!

The third is World Models.  Ha & Schmidhuber's 2018 paper was accompanied by a blog post that was both interactive and full of moving images. 

Alongside this outstanding presentation sits a body of work that is presentated in a number of blog posts.

The technical work of World Models uses supervised and unsupervised learning providing representations to an evolutionary controller.

I had never worked with any of the techniques used in World Models. The influence of me learning these techniques has been most visible in the projects of my students at Data Science Retreat.  Shout out to Mack (MDN) Samson (VAE) and Stas (World Models + PPO).

## The promise of learning a model of the world

The other impression of the World Models work is the tagline *learning within a dream*.  This was something very relevant to work I was doing as an energy data scientist.  We had no simulator, and struggled to learn an environment model from limited amounts of customer data.

Being able to learn an environment model is a super power in any control problem.  Note that you can replace learn with approximate :)

World Models is an example of strong technical work presented well.

So what is in a World Models agent?

# The Agent

F1 (whiteboard)

The World Models agent has three components a **vision**, **memory** and **controller**.

The key idea in World Models is **compression**.  Compression (also known as dimensionality reduction) is perhaps the fundamental operation in machine learning.

Why is dimensionality reduction valuable?  **Decisions are easier in low dimensional spaces.  Prediction is also easier**!

World Models uses compression in two components.  The vision component compresses a high dimensional observation of the environment $x$ to a low dimensional representation $z$.  This low dimensional representation is used as one of the two inputs to the controller.

The memory component predicts the next latent state $z'$ from the current latent state $z$.  The agent never uses the predicted next latent state $z'$ (which represents only one step in the future) but instead the hidden state of the LSTM used to predict $z'$.

It is curious that our agent never uses the final output of either the vision or the memory in the controller.  For both components the agent makes use of internal, compressed representations of either space or time.

Keeping the dimensionality reduction compression away from the controller allows us to use a simple linear controller with an evolutionary algorithm.  A major benefit of an evolutionary method is the ability to parallelize rollouts, which makes up for learning from a weaker signal (total episode reward) than more sample efficient model free or model based reinforcement learning.

# The Environment

The environment our agent will interact with 

I used the same version of OpenAI gym as the paper (`gym==0.9.4`).

## Markov Decision Process

A Markov Decision Process (MDP) is a framework for decision making.  It can be defined as:

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma) $$

- set of states $\mathcal{S}$
- set of actions $\mathcal{A}$
- set of rewards $\mathcal{R}$
- state transition function $ P(s'|s,a) $
- reward transition function $ R(r|s,a,s') $
- distribution over initial states $d_0$
- discount factor $\gamma$

It is common to make the distinction between the state $s$ and observation $x$.  The state represents the true state of the environment, and has the Markov property.  The observation is what the agent sees.  The observation is less informative, and often not Markovian.

## `car-racing-v0` as a Markov Decision Process

In the `car-racing-v0` environment, the agents **observation** is raw image pixels (96, 96, 3) - this is cropped and resized to (64, 64, 3).

<center>
	<img src="/assets/world-models/f1-final.png">
	<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

In DQN four environment observations are stacked to make the observation more Markov.  This isn't done in World Models - the observation is only of a single frame.

The **action** has three continuous dimensions - `[steering, gas, break]` (could give min / max).  A continuous action space can make some reinforcement learning 

The **reward** function is -0.1 for each frame, +1000 / N for each tile visited (N = total tiles on track).  This reward function encourages quick driving forward on the track.

Of particular interest is a hyperparameter controlling the episode length (known more formally as the horizon).  This is set to 1,000 throughout the paper codebase.  Changing this can have interesting effects on agent performance. (Horizion)

## Working with `car-racing-v0`

Of particular importance is using `env.viewer.window.dispatch_events()`  in the `reset()` and `step()` methods ([see the GitHub issue here](https://github.com/openai/gym/issues/976)).  If you don't use it you will get corrupt environment observations!

<center>
	<img src="/assets/world-models/corrupt.jpeg">
	<figcaption>Corrupt</figcaption>
  <div></div>
</center>

```python
#  do this before env.step(), env.reset()
self.viewer.window.dispatch_events()
```

Notebook about resizing-obserwation.ipynb (has hardcoded path to an observation


Instability when parallelizing over CMAES

Below the code for sampling environment observations is given in full ([see the source here](https://github.com/ADGEfficiency/world-models-dev/blob/master/worldmodels/dataset/car_racing.py)):

```python
# worldmodels/dataset/car_racing.py
# worldmodels/dataset/*
```

# Vision

## Why do we need to see?

Vision is **dimensionality reduction**.  It is the process of reducing the high dimensional image data into a lower dimensional space.

A canonical example in modern computer vision is image classification, where an image can be mapped throughout a convolutional neural network to a low dimensional space (cat or dog).

Another would be the flight or fight response.

In our `car-racing-v0` environment, a low dimensional representation of the environment observation might include:

```python
observation = [
	on_road = 1,
	corner_to_the_left = 1,
	corner_to_the_right = 0
]
```

Using these three numbers, we could imagine deriving a simple control policy.  Try to do this with 27,648 numbers arranged in a shape (96, 96, 3)!

We need to see in order to reduce high dimensional data

Vision is dimensionality reduction.  How does our agent see?

## What the controller sees (move up?)

The vision of the World Models agent reduces the environment observation $x$ (96, 96, 3) into a low dimensional representation $z$ (32,) known as the **latent space**.

The latent representation is **hidden**.  It is unobserved - we have no labels for these 32 variables.  The controller uses the latent representation $z$ as one of its inputs.  It never uses $x$ (or it's reconstruction $x'$) directly.

How do we learn this latent representation if we don't have examples?  One technique is a Variational Autoencoder.

## The Variational Autoencoder

A **Variational Autoencoder (VAE)** forms the vision of our agent.

The VAE is a **generative** model that learns the data generating process.  The data generating process is $P(x,z)$ - the joint distribution over our data (the probability of $x$ and $z$ occurring together).

But what is a generative model?

## Generative versus discriminative models

All approaches in supervised learning are either generative or discriminative.

### Generative models

**Generative models learn a joint distribution** $P(x, z)$ (the probability of $x$ and $z$ occurring together).  Generative models generate new, unobserved data $x'$.

We can derive this process for generating new data, from the definition of conditional probability:

$$ P(x \mid z) = \frac{P(x, z)}{P(z)} $$

Rearranging this definition gives us a decomposition of the joint distribution. This is the product rule of probability:

$$P(x, z) = P(x \mid z) \cdot P(z)$$

This decomposition describes the entire generative process.  First sample a latent representation:

$$z \sim P(z)$$

Then sample a generated data point $x'$, using the conditional probability $P(x \mid z)$:

$$x' \sim P(x \mid z)$$

These sampling and decoding steps only describe the generation of new data $x'$ from an unspecified generative model.  It doesn't describe the entire structure of a VAE.

### Discriminative models

Unlike generative models, **discriminative models learn a conditional probability** $P(x \mid z)$ (the probability of $x$ given $z$).  Discriminative models predict, using observed $z$ to predict $x$.  This is simpler than generative modelling.

A common discriminative computer vision problem is classification, where a high dimensional image is fed through convolutions and outputs a predicted class.

Now we understand the context of generative models in supervised learning, we can look at the context of the VAE within generative models.


Where to generative models fit in the context of other supervised learning methods?


### The VAE in context

The VAE sits alongside the Generative Adversarial Network (GAN) as the state of the art in generative modelling.  

The figure below shows the outstanding progress in image quality generated by GANs.  GANs typically outperform VAEs on reconstruction quality, with the VAE providing better support over the data (support meaning the number of different values a variable can take).

<center>
	<img src="/assets/world-models/gan.png">
	<figcaption>
		Progress in GANS - <a href="https://www.iangoodfellow.com/slides/2019-05-07.pdf">Adverserial Machine Learning - Ian Goodfellow - ICLR 2019</a>
	</figcaption>
  <div></div>
</center>

The VAE has less in common with classical (sparse or denoising) autoencoders, which both require the use of the computationally expensive Markov Chain Monte Carlo.

### What makes the VAE a good choice for the World Models agent?

A major benefit of generative modelling is the ability to generate new samples $x'$.  Yet our World Models agent never uses $x'$ (whether a reconstruction or a new sample).

The role of the VAE in our agent is to provide a compressed representation $z$ by learning to encode and decode a latent space.

The lower dimensional latent space is easier for our memory and controller to work with.

What qualities do we want in our latent space?  One is **meaningful grouping**.  This requirement is a challenge in traditional autoencoders, which tend to learn spread out latent spaces.

Meaningful grouping means that similar observations exist in the same part of the latent space, with samples that are close together in the latent space producing similar images when decoded.  This grouping means that even observations that the agent hadn't seen before could be responded to the same way.

Meaningful grouping also allows **interpolation** - meaning that we can understand observations we haven't seen before, if they are encoded close to observations we have seen before.

So how do we get meaningful encoding of an observation?  One technique is to constrain the size of the latent space (32 variables for the World Models VAE).  The VAE puts even more effort into TODO

## VAE structure

The VAE is formed of three components - an encoder, a latent space and a decoder.  But before we discuss these, we need to introduce convolution.

f2

### Convolution

### Encoder

The primary function of the encoder is **recognition**.  The encoder is responsible for recognizing and encoding the hidden **latent variables**.

The encoder is built from convolutional blocks that map from the input image ($x$) (64, 64, 3) to statistics (means & variances) of the latent variables (length 64 - 2 statistics per latent space variable).

### Latent space

Constraining the size of the latent space (length 32) is one way auto-encoders learn efficient compression of images.  All of the information needed to reconstruct a sample $x$ must exist in only 32 numbers!

The statistics parameterized by the encoder are used to form a distribution over the latent space - a diagonal Gaussian.  A diagonal Gaussian is a muntivariate Gaussian with a diagonal covariance matrix.  This means that each variable is independent.

(is this enforcing a gaussian prior or posterior?)

This parameterized Gaussian is an approximation.  Using it will limit how expressive our latent space is,

$$z \sim P(z \mid x)$$

$$ z \mid x \approx \mathbf{N} \Big(\mu_{\theta}, \sigma_{\theta}\Big) $$

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
- decode the sampled latent space into a reconstructed image $x' \sim D_{\theta}(x' \mid z)$

### Generation

$z$ -> $x'$

- sample a latent space $z \sim P(z)$
- decode the sampled latent space into a reconstructed image $x' \sim D_{\theta}(x' \mid z)$

```python
# worldmodels/vision/vae.py
```

## The backward pass

The VAE uses **likelihood maximization** to learn this joint distribution $P(x,z)$.  Likelihood maximization is the process of maximizing the similarity between two distributions.  In our case these distributions are the distribution over our training data (the data generating process) and our parametrized approximation (a convolutional neural network).

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

$$ \mathbf{ELBO}(\theta, \theta) = \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta}(x' \mid z) \Big] - \mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z) \Big) $$

Note the appearance of our decoder $D_{\theta}(x \mid z)$.  The decoder is used to approximate the true posterior $P(x' \mid z)$ - the conditional probability distribution over the reconstruction of latent variables into a generated $x'$ (given $x$).

The last step is to convert this $\mathbf{ELBO}$ maximization into a more familiar loss function minimization.  We now have the VAE loss function's final mathematical form - in all it's tractable glory:

$$ \mathbf{LOSS}(\theta) = - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta} (x' \mid z) \Big] + \mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big)  $$

The loss function has two terms - the log probability of the reconstruction (aka the decoder) and a $\mathbf{KLD}$ between the latent space (sampled from our encoder) and the latent space prior $P(z)$.

Remember that the loss function above is the result of minimizing the $\mathbf{KLD}$ between our encoder $E_{\theta}(z \mid x)$ and the data generating distribution $P(z \mid x)$.  What we have is a result of maximizing the log-likelihood of the data.

## Implementing the loss function in code

Although our loss function is in it's final mathematical form, we will make three more modifications before we implement it in code:
- convert the log probability of the decoder into a pixel wise reconstruction loss
- use a closed form solution to the $\mathbf{KLD}$ between our encoded latent space distribution and the prior over our latent space $P(x)$
- refactor the randomness using reparameterization

We will look at these two modifications in terms of the two terms in our loss function (??)

### First term - reconstruction loss

$$ - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta} (x' \mid z) \Big] $$

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

$$ \mathbf{LOSS}(\theta) = - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta} (x' \mid z) \Big] + \mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big)  $$

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

## Why do we remember?

For a human, memory has many roles.

For our World Models agent, the agent remembers past transitions in order to predict future transitions.

## The World Models memory

The memory of the World Models agent is a discriminative model, that models the conditional probability:

$$ P(z' | z, a) $$

The primary role of the memory in the World Models agent is compression of the future into a low dimensional representation $h$.

This low comression of time $h$ is the hidden state of an LSTM.

The LSTM to trained only to predict the next latent vector $z'$, but learns a longer representation of time via hidden state (specifically $h$)

The memory's life is made easier by being able to predict in the low dimensional space learnt by the VAE.

The memory has two components - an LSTM and a Gaussian Mixture head.  Both these together form a mixed density network.

##  Gaussian Mixtures

A Gaussian mixture can model arbitrary conditional probability distributions.  It requires no assumption of individual variables in the prediction, even though we learn a diagonal covariance (do we???).

In a more general setting, the variances learnt by a Gaussian mixture can be used as a measure of uncertainty.

A primary motivation behind using a mixture of distributions is that we can approximate **multi-modal** distributions.

Probability distribution output by a mixture can (in principle!) be calculated.  The flexibility is similar to a feed forward neural network, and likely has the same distinction between being able approximate versus being able to learn.


Bishop (?) shows that by training a neural network using a least squares loss function, we are able to learn two statistics.  One is the conditional mean, which is our prediction.  The second statistic is the variance, which we can approximate from the residual.

We can use these two statistics to form a Gaussian.

Our kernel of choice is the Gaussian, which has a probability density function:

$$ \phi (z' \mid z, a) = \frac{1}{\sqrt{(2 \pi) \sigma(z, a)}} \exp \Bigg[ - \frac{\lVert z' - \mu(z, a) \rVert^{2}}{2 \sigma(z, a)^{2}} \Bigg] $$

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

*For a deeper look at LSTM's, I cannot reccomend the blog post [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) highly enough.*

The motivation for using an LSTM to approximate the transition dynamics of the environment is that an LSTM is a **recurrent neural network**.  In the `car-racing-v0` environment the data is a sequence of latent representations $z$ of the observation

Recurrent neural networks process data in a sequence:

$$ P(x' | x, h) $$

Where $h$ is the hidden state of the recurrent neural network.



The LSTM was introduced in 1997 by Hochreiter & Schmidhuber.  A key contribution of the LSTM was overcoming the challenge of long term memory with only a single representation of the future.

The LSTM is a recurrent neural network, that makes predictions based on the following:

$$ P(x' | x, h, c) $$ 

Where $h$ is the hidden state and $c$ is the cell state.  Using two variables for the LSTM's internal representation allows the LSTM to learn both a long and short term representation of the future.

The long term representation is the **cell state** $c$.  The cell state is an information superhighway.

The short term representation is the **hidden state** $h$.

Sigmoid often used as an activation for binary classification.  For LSTMs, we use the sigmoid to control infomation flow.

Tanh is used to generate data.  Neural networks like values in the range -1 to 1, which is exactly how a tanh generates data (with some non-linearity in between).

Infomation is added or removed from both the cell and hidden states using gates.

The gates are functions of the hidden state $h$ and the data $x$.

GET, PUT, DELETE - Create, read, update, delete

<center>
	<img src="/assets/world-models/lstm.png">
<figcaption> The LSTM - from [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)</figcaption>
</center>

### Forget gate

The first gate is the **forget gate**.  The forget gate works like the `DELETE` request in an REST API.

The forget gate multiplies the cell state by a sigmoid.  A gate value of 0 would mean forget the entire cell state.  A gate value of 0 would mean remember the entire cell state.

The sigmoid used to control the forget gate is a function of the hidden state $h$ and the data $x$.

### Input gate

The second gate is the **input gate**.  The input gate works like a `PUT` or `POST` request.

This gate determines how we will update the cell state from $c$ to $c'$.  The infomation added to the cell state is formed from a sigmoid (that controls which values to update) and a tanh (that generates the new values).

### Output gate

The final gate determines what the LSTM outputs.  This gate works like a `GET` request.

A sigmoid (based on the hidden state $h$) determines which parts of the cell state we will output.  This sigmoid is applied to the updated cell state $c'$, after the updated cell state $c'$ was passed through a tanh layer.

```python
#worldmodels/memory/memory.py

#worldmodels/memory/train_memory.py
```

## Putting the Memory together

$$ M_{\theta}(z'| z, a, h, c) $$

## Implementing the Memory in code

From a software development perspective, development of the `Memory` class was done in two distinct approaches.

### Performance based testing

The first was testing the generalization of the MDN on a toy dataset.  The inspiration and dataset came directly from the excellent []().

To isolate any bugs I first tested the Gaussian mixture head with a fully connected net.

![]()

The next test was with an LSTM generating the statistics of the mixture:

![]()

### Unit testing

The performance based testing was combined with lower level unit testing:

```python

```

### Training results


```python
# mem, train mem

```

# Control

http://blog.otoro.net/2017/10/29/visual-evolution-strategies/

https://arxiv.org/pdf/1604.00772.pdf

$$ C_{\theta}(a' \mid z, h) $$

## Why do we need control?

The vision and memory components we have looked at so far are provide a compressed representation of the current and future environment observations.

We need a controller to take these representations and select an action.

The World Models agent uses a simple linear function for control.  The parameters of this function are learnt using the Covariance Matrix Adapation Evolution Stragety (CMA-ES).

CMA-ES is an evolutionary algorithm.  We can place this learning algorithm in context using a concept from ?.

## The four competences

Talk about four competences

only learn total ep rew, gradient free, fitness & objective fctn

Less data efficient, less computation

## Evolutionary algorithms

> Randomized search algorithms are regarded to be robust in a rugged search landscape,which can comprise discontinuities, (sharp) ridges, or local optima. The covariance matrixadaptation (CMA) in particular is designed to tackle, additionally, ill-conditioned and nonseparable problems
https://arxiv.org/pdf/1604.00772.pdf

Evolutionary algorithms sit right at the bottom of our four competences.  This position is not derogatory - the learning process of evolution is responsible for everything around you.

Computational evolutionary methods can be used for non-linear, non-convex, gradient free optimization.

Rather that peeking into the temporal structure of a Markov Decision Process, the evolutionary method favours a more general **black box search**.

Evolutionary methods are often stochastic

Generate, test, select

```python
for generation
	generate 
	test
	select
```

Genetic would include operations such as recombination by crossover or mutation in the generate step.

Now that we understand the context of evolutionary methods, let's look at the method used by the controller - CMA-ES.

## CMA-ES

Update mean using n best

Learn the full covariance of the parameter space

Rank mu = uses info from entire population

Rank one = info of correlation between generations.

Step size control = TODO

VAE = only diagonal, mix = diagonal.  Here we work with something much more significant.

Overcomes typical problems with evolutionary methods:
- poor performance on badly scaled / non-separable problems by combining rank one & rank mu update of C
- prevent degeneration with small population sizes (rank 1 & mu)
- premature convergence prevented by step size control

Ha reccomends limiting to 10k parameters, as the covariance matrix calculation is $O(n^{2})$.

### Generate

Sample

$$ x \sim \mu + \sigma \mathbf{N} \Big(0, C \Big) $$

### Test

Total episode reward

### Select

Selecting n best

N best -> statistics

$$ \mu_{g+1} = \frac{1}{N_best} \sum_{N_{best}} x_{g} $$

Estimating a covariance matrix

Important to remember which covariance matrix we want to estimate - a covariance matrix that will generate good parameters for our controller.

Let's imagine we have a parameter space with two variables, $x$ and $y$.  We can estimate the statistics needed for a covariance matrix:

$$ \mu_{x} = \frac{1}{N} \sum_{pop} x $$

$$ \mu_{y} = \frac{1}{N} \sum_{pop} y $$

$$ \sigma^{2}_{x} = \frac{1}{N} \sum_{pop} \Big( x - \mu_{x} \Big)^{2} $$

$$ \sigma^{2}_{y} = \frac{1}{N} \sum_{pop} \Big( y - \mu_{y} \Big)^{2} $$

$$ \sigma_{xy} = \frac{1}{N} \sum_{pop} \Big( x - \mu_{x} \Big) \Big( y - \mu_{y} \Big) $$

$$\mathbf{COV} = \begin{bmatrix}  \sigma^{2}_{x} & \sigma_{xy} \\ \sigma_{yx} &  \sigma^{2}_{y}\end{bmatrix}$$

These statistics will are an approximation of the true statistics used to generate (sample?) the data $x$ and $y$.

In CMA-ES we select the $N_{best}$ parameters from generation $g$.

We calculate the mean used to generate the next generation $g+1$ using the sample average across our $N_{best}$ parameters:

But we do not use this mean $g+1$ when we form our covariance matrix, instead we use the sample mean from the current generation $g$:

$$ $$

Estimating the $COV$ from a single generation is unreliable.  A more reliable method uses adapation (rank mu)

Rank mu = uses info from entire population

Rank one = info of correlation between generations.

Step size control = TODO

```python
for population in range(populations):
	parameters = solver.ask()
	fitness = environment.evalute(parameters)
	solver.tell(parameters, fitness)
```

## Implementing the controller

Careful where you import


```python
#worldmodels/control
```

# Timeline

# Methods

Step by step to reproduce

Bash script to dl weights

# Final results

# Discussion

# Refereces

(same as ToC)

ref dqn, rainbow
