---
title: 'World Models'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: Ha & Schmidhuber's World Models (2018) reimplemented in Tensorflow 2.0 

---

<center>
	<img src="/assets/world-models/f0.gif">
<figcaption>Performance of the final agent on a conveniently selected random seed. The cumulative episode reward is shown in the lower right.  This agent & seed achieves 893. 900 is solved.</figcaption>
</center>

# Table of Contents

* [Table of Contents](#table-of-contents)
* [Key Resources](#key-resources)
* [Motivations](#motivations)
	* [Why reimplement a paper?](#why-reimplement-a-paper)
	* [Why reimplement World Models?](#why-reimplement-world-models)
		 * [DQN](#dqn)
		 * [AlphaGo Zero](#alphago-zero)
		 * [World Models](#world-models)
	* [The promise of learning a model of the world](#the-promise-of-learning-a-model-of-the-world)
	* [The four competences](#the-four-competences)
		 * [1 - Darwinian](#1---darwinian)
		 * [2 - Skinnerian](#2---skinnerian)
		 * [3 - Popperian](#3---popperian)
		 * [4 - Gregorian](#4---gregorian)
* [The Agent](#the-agent)
	* [Compression](#compression)
* [The Environment](#the-environment)
	* [Markov Decision Process](#markov-decision-process)
	* [car-racing-v0 as a Markov Decision Process](#car-racing-v0-as-a-markov-decision-process)
	* [Working with car-racing-v0](#working-with-car-racing-v0)
* [Vision](#vision)
	* [Why do we need to see?](#why-do-we-need-to-see)
	* [What the controller sees (move up?)](#what-the-controller-sees-move-up)
	* [The Variational Autoencoder](#the-variational-autoencoder)
	* [Generative versus discriminative models](#generative-versus-discriminative-models)
		 * [Generative models](#generative-models)
		 * [Discriminative models](#discriminative-models)
		 * [The VAE in context](#the-vae-in-context)
		 * [What makes the VAE a good choice for the World Models agent?](#what-makes-the-vae-a-good-choice-for-the-world-models-agent)
	* [VAE structure](#vae-structure)
		 * [Convolution](#convolution)
		 * [Encoder](#encoder)
		 * [Latent space](#latent-space)
		 * [Decoder](#decoder)
	* [The three forward passes](#the-three-forward-passes)
		 * [Compression](#compression-1)
		 * [Reconstruction](#reconstruction)
		 * [Generation](#generation)
	* [The backward pass](#the-backward-pass)
	* [Implementing the loss function in code](#implementing-the-loss-function-in-code)
		 * [First term - reconstruction loss](#first-term---reconstruction-loss)
		 * [Second term - regularization](#second-term---regularization)
		 * [Reparameterization trick](#reparameterization-trick)
	* [Vision - Summary](#vision---summary)
* [Memory](#memory)
	* [Why do we remember?](#why-do-we-remember)
	* [The World Models memory](#the-world-models-memory)
	* [Gaussian Mixtures](#gaussian-mixtures)
	* [LSTM](#lstm)
		 * [Forget gate](#forget-gate)
		 * [Input gate](#input-gate)
		 * [Output gate](#output-gate)
	* [Putting the Memory together](#putting-the-memory-together)
	* [Implementing the Memory in code](#implementing-the-memory-in-code)
		 * [Performance based testing](#performance-based-testing)
		 * [Unit testing](#unit-testing)
		 * [Training results](#training-results)
* [Control](#control)
	* [Why do we need control?](#why-do-we-need-control)
	* [Evolution](#evolution)
		 * [Darwinian compe](#darwinian-compe)
	* [Computational evolution](#computational-evolution)
		 * [$(1, lambda)$-ES](#1-lambda-es)
		 * [General purpose optimization](#general-purpose-optimization)
		 * [Sample inefficiency](#sample-inefficiency)
		 * [Parallel rollouts](#parallel-rollouts)
	* [ADGEfficiency/evolution](#adgefficiencyevolution)
	* [CMA-ES](#cma-es)
		 * [Generate](#generate)
		 * [Test](#test)
		 * [Select](#select)
		 * [Estimating a covariance matrix](#estimating-a-covariance-matrix)
	* [Rank-one update](#rank-one-update)
	* [Implementing the controller](#implementing-the-controller)
* [Timeline](#timeline)
	* [Iterative training procedure](#iterative-training-procedure)
* [Methods](#methods)
* [Final results](#final-results)
* [Discussion](#discussion)
* [Refereces](#refereces)

# Key Resources

- reimplementation code base - [ADGEfficiency/world-models](https://github.com/ADGEfficiency/world-models)
- original paper code base - [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)
- [interactive blog post](https://worldmodels.github.io/) - World Models
- [2018 paper](https://arxiv.org/pdf/1809.01999.pdf) - Recurrent World Models Facilitate Policy Evolution 
- [2018 paper](https://arxiv.org/pdf/1803.10122.pdf) - World Models 
- [David Ha talk at NIPS](https://youtu.be/HzA8LRqhujk) - Recurrent World Models Facilitate Policy Evolution
- extensive collection of resources - [ADGEfficiency/rl-resources/world-models](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models)

# Motivations

My main side project in 2019 was a reimplementation of the 2018 paper by Ha & Schmidhuber.  

This project dominated any spare time I had - usually one to three days per month.

<center>
	<img src="/assets/world-models/commits-month.png">
	<figcaption>Commits per month.  Not all commits are made equal.</figcaption>
  <div></div>
</center>

TODO AWS spend

## Why reimplement a paper?

The original idea for reimplementing a paper came from reading an Open AI job advertisement.  Seeing a tangible goal that could put me in the ballpark of a world class machine learning lab, I set out looking for a paper to reimplement.

I specifically remember reading *high quality implementation*.  This echoed in my mind as I developed the project.

## Why reimplement World Models?

World Models was the third of three papers that have blown me away in the three years I have been working with machine learning.  

### DQN

The first blew me away without me even reading it.  I have a memory of seeing a YouTube video DQN playing the Atari game Breakout.  Even though I knew nothing of reinforcement learning, the significance of a machine could learn to play a video game from pixels was made clear.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/V1eYniJ0Rnk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
</center>

I had no way of knowing that the algorithm I was watching would be one I implement four times, or that I would teach the mechanics of DQN and it's evolution into Rainbow over twenty times.

### AlphaGo Zero

The second was AlphaGo Zero.  The publication of AlphaGo Zero in October 2017 came out after I had taught my course on reinforcement learning twice.

<center>
	<img src="/assets/world-models/Zero_act_learn.png">
	<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

At this stage (9 months after my transition into data science), I didn't fully grasp all of the mechanics in AlphaGo.  But I knew enough to understand the significance of the changes - *tabula rasa* learning among the most important.

### World Models

<center>
	<img src="/assets/world-models/ha-blog.png">
<figcaption>from https://worldmodels.github.io/ </figcaption>
</center>

The third is World Models.  Ha & Schmidhuber's 2018 paper was accompanied by a blog post that was both interactive and full of moving images. 

Alongside this outstanding presentation sits a body of work that is presentated in a number of blog posts.

The technical work of World Models uses supervised and unsupervised learning providing representations to an evolutionary controller.

I had never worked with any of the techniques used in World Models. The influence of me learning these techniques has been most visible in the projects of my students at Data Science Retreat.  Shout out to Mack (MDN) Samson (VAE) and Stas (World Models + PPO).

## The promise of learning a model of the world

We operate in a world with spatial and temporal dimensions.  A *world model* is an abstract representation of these dimensions.



The other impression of the World Models work is the tagline *learning within a dream*.  This was something very relevant to work I was doing as an energy data scientist.  We had no simulator, and struggled to learn an environment model from limited amounts of customer data.

Being able to learn an environment model is a super power in any control problem.  Note that you can replace learn with approximate :)

World Models is an example of strong technical work presented well.

So what is in a World Models agent?

## The four competences

I read Daniel Dennet's *From Bacteria to Bach and Back* in 2018.  Dennet introduces the four grades of competence - competence being the ability to act well.

This is a key part of my teaching, as it puts in context evolutionary (Darwinian), model-free reinforcement learning (Skinnerian) and model-based reinforcement learning (Popperian).  I do not know of a computational method that builds it's own thinking tools.

### 1 - Darwinian
- pre-designed and fixed competence
- no learning within lifetime
- global improvement via local selection

### 2 - Skinnerian
- the ability to adjust behaviour through reinforcement
- learning within lifetime
- hardwired to seek reinforcement

### 3 - Popperian
- learns models of the environment
- local improvement via testing behaviours offline
- crows, cats, dogs, dolphins & primates

### 4 - Gregorian
- builds thinking tools
- arithmetic, democracy & computers
- systematic exploration of solutions
- local improvement via higher order control of mental searches
- only humans

# The Agent

F1 (whiteboard)

The World Models agent has three components a **vision**, **memory** and **controller**.

The vision and memory components learn a low dimensional representation of the environment observation.

## Compression

Dimensionality reduction is a fundamental operation in machine learning.  

Why is dimensionality reduction valuable?  
- decisions are easier in low dimensional spaces
- prediction is also easier

Dimensionality reduction can be thought of as a from of lossless compression of infomation from a higher dimensional space into a lower dimensional space.

World Models uses compression in two components.  The vision component compresses a high dimensional observation of the environment $x$ to a low dimensional representation $z$.  This low dimensional representation is used as one of the two inputs to the controller.

The memory component predicts the next latent state $z'$ from the current latent state $z$.  The agent never uses the predicted next latent state $z'$ (which represents only one step in the future) but instead the hidden state of the LSTM used to predict $z'$.  The low dimensional LSTM hidden state is the second input to the controller.

It is curious that our agent never uses the final output of either the vision or the memory in the controller.  For both components the agent makes use of internal, compressed representations of either space or time.

Keeping the dimensionality reduction compression away from the controller allows us to use a simple linear controller to map to an action.  Using a low capacity controller (784 parameters ???) allows using an evolutionary algorithm to find parameters of a good policy.

# The Environment

Our agent interacts with the `car-racing-v0` environment from OpenAI's `gym` library.  I used the same version of `gym` as the paper (`gym==0.9.4`).

## Markov Decision Process

We can describe the `car-racing-v0` environment as a Markov Decision Process.  A Markov Decision Process (MDP) is a framework for decision making.  It can be defined as:

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma, H) $$

- set of states $\mathcal{S}$
- set of actions $\mathcal{A}$
- set of rewards $\mathcal{R}$
- state transition function $ P(s' \mid s,a) $
- reward transition function $ R(r \mid s,a,s') $
- distribution over initial states $d_0$
- discount factor $\gamma$
- horizion $H$

It is common to make the distinction between the state $s$ and observation $x$.  The state represents the true state of the environment, and has the Markov property.  The observation is what the agent sees.  The observation is less informative, and often not Markovian.

Because the World Models agent uses the total episode reward as a learning signal, there is also no role for a discount rate $\gamma$.

## `car-racing-v0` as a Markov Decision Process

In the `car-racing-v0` environment, the agents **observation** is raw image pixels (96, 96, 3) - this is cropped and resized to (64, 64, 3).

Temporal structure

<center>
	<img src="/assets/world-models/f1-final.png">
	<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

In DQN four environment observations are stacked to make the observation more Markov.  This isn't done in World Models - the observation is only of a single frame.

The **action** has three continuous dimensions - `[steering, gas, break]` (could give min / max).  A continuous action space can make some reinforcement learning 

The **reward** function is -0.1 for each frame, +1000 / N for each tile visited (N = total tiles on track).  This reward function encourages quick driving forward on the track.

The **horizon** (aka episode length) is set to 1,000 throughout the paper codebase.  Changing this can have interesting effects on agent performance, and I wonder how significant this hyperparameter is on final agent performance.

## Working with `car-racing-v0`

Of particular importance is using `env.viewer.window.dispatch_events()` in the `reset()` and `step()` methods of the environment ([see the GitHub issue here](https://github.com/openai/gym/issues/976)).  If you don't use it you will get corrupt environment observations!

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

The $\mathbf{ELBO}$ is given as the expected difference in log probabilities when we are sampling our latent vectors from our encoder $E_{\theta}(z \mid x)$:

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

Note that even when the memory has the ability to think 'long-term', this still pales in comparison to the long term memorpwe posess.  The authors note that a higher caparity external memory is needed for explorinc moce complex worlds.

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

$$ C_{\theta}(a' \mid z, h) $$

The final component of the World Models agent is the controller. 

The vision and memory components provide a compressed representation of the current and future environment observations.  

The controller is a simple linear function that maps from these compressed representations ($z$ and $h$) to an action $a$.

The algorithm used by the World Models agent for finding the controller parameters is Covariance Matrix Adapation Evolution Stragety (CMA-ES).

## Why do we need control?

One task required by agents in a Markov Decision Process is credit assignment - determining which actions lead to reward.

An agent that understands credit assignment can use this understanding to create a policy.

In reinforcement learning, credit assignment is often learnt by a function that also learns other tasks, such as learning a low dimensional representation of a high dimensional image.  

In DQN, the action-value function learns to both extract features from the observation and map them to an action.

In the World Models agent these tasks are kept separate, with vision responsible for learning a spatial representation and the memory learning a temporal representation.

This separation allows the use of a simple linear controller, compeletly dedicated to learning how to assign credit.

Another benefit of having a simple, low parameter count controller is that it opens up less sample efficient but more general methods for finding the model parameters.

The downside of this is that our vision and memory might use capacity to learn features that are not useful for control, or not learn features that are useful.

## Evolution

Evolutionary learning is the driving force in our universe.  At the heart of evolution is a paradox - that failure at a low level leads to improvement at a high level.  Examples include:
- biology = survival
- business = money
- training neural nets using backpropagation = error / residual

Evolution is iterative improvement using a generate, test, select loop:

```
for generation
  generate 
  test
  select
```

In the **generate** step a population is generated, using infomation from previous steps.

In the **test** step, the population interacts with the environment, and is assigned a single number as a score.  

In the **select** step, members of the current generation are selected (based on their fitness) to be used to generate the next step.

There is so much to learn from this evolutionary process:
- failure at a low level driving improvement at a higher level
- the effectivness of iterative improvement
- the requirement of a dualistic (agent and environment) view

In particular, understanding that failure leads to improvement is an important lesson.

We now have an understanding of the general process of evolutionary learning.  Now let's look at how we do this *in silico*.

### Darwinian compe

Evolutionary algorithms sit right at the bottom of our four competences.  This position is not derogatory - the learning process of evolution is responsible for everything around you.  Sample inefficiency isn't a problem over millions  of years.

It is an example of Darwinian improvement, inspired by biological evolution, the results of which we see today.

## Computational evolution

Computational evolutionary algorithms are inspired by biological evolution.  They perform non-linear, non-convex and gradient free optimization.  Evolutionary methods can deal with the challenges that discontinuities, noise, outliers and local optima pose in optimization.

Computational evolutionary algorithms are often the successive process of sampling parameters of a function (i.e. a neural network) from a distribution.  This process can be further extended by other biologically inspried mechanics, such as crossover or mutation - knownn as genetic algorithms.

A common Python API for computation evolution is the **ask, evaluate and tell** loop.  This can be directly mapped onto the generate, test & select loop introduced above:

```python
for population in range(populations):
  #  generate
  parameters = solver.ask()
  #  test
  fitness = environment.evalute(parameters)
  #  select
  solver.tell(parameters, fitness)
```

### $(1, \lambda)$-ES

The simplest evolution strategy (ES) is $(1, \lambda)$-ES:

$$ \theta \sim N(\mu, I) $$

$$ FITNESS = ENV(\theta) $$

$$ \theta_{best} = \text{argmin}_{\theta} {FITNESS} $$

$$ \mu = \frac{1}{N_{best}} $$


From a practical standpoint, the most important features of evolutionary methods are:
- general purpose optimization
- poor sample efficiency
- parallelizability

### General purpose optimization
> Randomized search algorithms are regarded to be robust in a rugged search landscape,which can comprise discontinuities, (sharp) ridges, or local optima. 
Evolutionary algorithms learn from a single number per generation - the total episode reward.  This single number serves as a measurement of a population's fitness.  

This is why evolutionary algorithms are **black box** - unlike less general optimizers they don't learn from the temporal structure of the MDP.  They are also gradient free.

### Sample inefficiency

How sample efficient an algorithm is depends on how much experience (measured in transition between states in an MDP) an algorithm needs to achieve a given level of performance.  

It is of key concern if compute is purchased on a variable cost basis.

The causes of sample inefficiency are
- noisy signal
- weak signal

Working in opposition to these are:
- less computation per episode
- ability to parallelize

### Parallel rollouts

A key feature of Darwinian learning is fixed competence, with population members not learning within lifetime.  This feature means that each population member can learn independently, and hence be parallelized.

This is a major benefit of evolutionary methods, which helps to counteract their sample inefficiency.

## `ADGEfficiency/evolution`

I hadn't worked with evolutionary algorithms before this project.  Due to my simple mind that relies heaivly on empiciral understanding, I often find implementing 

implemented $(1, \lambda)$-ES from scratch alongside a wrapper on `pycma` in a separate repo [ADGEfficiency/evolution]().

Below is the performance of the $(1, \lambda)$-ES algorithm on the ? problem.

Now that we understand the context of evolutionary methods, let's look at the method used by the controller - CMA-ES.

## CMA-ES

CMA-ES is the evolutionary algorithm used by our World Models agent to find good parameters of it's linear controller.

Ha suggests that CMA-ES is effective for up to 10k parameters, as the covariance matrix calculation is $O(n^{2})$.

A key feature of CMA-ES is the successive estimation of a full covariance matrix.  When combined with an estimation of the mean, these statistics can be used to form a mulitvariate Gaussian.

It is worth pointing out that the Gaussian's we parameterized in the vision and memory components have diagonal covariances, which mean each variable is independent of the changes in all the other variables.

Overcomes typical problems with evolutionary methods:
- poor performance on badly scaled / non-separable problems by combining rank one & rank mu update of C
- prevent degeneration with small population sizes (rank 1 & mu)
- premature convergence prevented by step size control

We can introduce CMA-ES in the context of generate, test and select.

### Generate

The generation step in CMA-ES involves sampling a population from a multivariate Gaussian:

$$ x \sim \mu + \sigma \mathbf{N} \Big(0, C \Big) $$

### Test

The test step in CMA-ES involves parallel rollouts of the population parameters in the environment.

### Select

The selection step in CMA-ES involves selecting the best $n$ members of the population.  These population members are used to update the statistics of our multivariate Gaussian.

We first update our estimate of the mean using a sample average over $n_{best}$ from the current generation $g$:

$$ \mu_{g+1} = \frac{1}{n_{best}} \sum_{n_{best}} x_{g} $$

Our next step is to update our covariance matrix $C$.  

### Estimating a covariance matrix

Let's imagine we have a parameter space with two variables, $x$ and $y$.  We can estimate the statistics needed for a covariance matrix:

$$ \mu_{x} = \frac{1}{N} \sum_{pop} x $$

$$ \mu_{y} = \frac{1}{N} \sum_{pop} y $$

$$ \sigma^{2}_{x} = \frac{1}{N-1} \sum_{pop} \Big( x - \mu_{x} \Big)^{2} $$

$$ \sigma^{2}_{y} = \frac{1}{N-1} \sum_{pop} \Big( y - \mu_{y} \Big)^{2} $$

$$ \sigma_{xy} = \frac{1}{N-1} \sum_{pop} \Big( x - \mu_{x} \Big) \Big( y - \mu_{y} \Big) $$

$$\mathbf{COV} = \begin{bmatrix}  \sigma^{2}_{x} & \sigma_{xy} \\ \sigma_{yx} &  \sigma^{2}_{y}\end{bmatrix}$$

The update of $C$ is done using the combination of two updates - rank-one and rank-$/mu$.

### Rank-one update

Above we have seen a general method of estimating a covariance matrix.  

In the context of our World Models agent, we might estimate the covarianre of our next population $g+1$ using our samples $x$ and taking a reference mean value from that population:

$$ \mathbf{COV}_{g+1} = \frac{1}{N_{best} - 1} \sum_{pop} \Big( x_{g+1} - \mu_{x_{g+1}} \Big) \Big( x_{g+1} - \mu_{x_{g+1}} \Big) $$

The approach used in a rank-one update instead uses a reference mean value from the previous generation $g$:

$$ \mathbf{COV}_{g+1} = \frac{1}{N_{best}} \sum_{pop} \Big( x_{g+1} - \mu_{x_{g}} \Big) \Big( x_{g+1} - \mu_{x_{g}} \Big) $$

Using the mean of the actual sample $\mu_{g+1}$ leads to an estimation of the covariance within the sample.   Using the mean of the previous generation $\mu_{g}$ leads to a covariance matrix that estimates the covariance of the sampled step.

### Rank-$\mu$ update

With the small population sizes required by CMA-ES, getting a good estimate of the covariance matirx using a ranksone update is challenging.

The rank-$\mu$ update uses a reference mean value that uses information from all previous generations.

This is done by taking an average over all previous estimated covariance matrices:

$$ \mathbf{COV} = \frac{1}{g+1} \sum_{gens} \frac{1}{\sigma^{2}} \mathbf{COV} $$










## Implementing the controller

Careful where you import

Run the 


```python
#worldmodels/control
```

# Timeline

## Iterative training procedure

Needed it!

Exploration problem

# Methods

Step by step to reproduce

Bash script to dl weights

# Final results

# Discussion

# Refereces

(same as ToC)

ref dqn, rainbow
