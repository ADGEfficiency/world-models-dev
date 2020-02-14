---
title: 'World Models'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: Ha & Schmidhuber's World Models (2018) reimplemented in Tensorflow 2.0.

---

todo
- summary section for v, m, c
- medium version (lessons learnt / summary post, no code)

use I!!!

*See the reimplementation code base here - [ADGEfficiency/world-models](https://github.com/ADGEfficiency/world-models).*

<center>
	<img src="/assets/world-models/f0.gif">
<figcaption>Performance of the final agent on a conveniently selected random seed. The cumulative episode reward is shown in the lower right.  This agent & seed achieves 893. 900 is solved.</figcaption>
</center>

# Table of Contents

# Key Resources

I keep an extensive collection of resources for the World Models paper and it's associated components at [ADGEfficiency/rl-resources/world-models](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models).

Important resources are below:
- my reimplementation code base - [ADGEfficiency/world-models](https://github.com/ADGEfficiency/world-models)
- original paper code base - [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments) - [advice on running the code base](http://blog.otoro.net/2018/06/09/world-models-experiments/) 
- [interactive blog post](https://worldmodels.github.io/) - [2018 paper](https://arxiv.org/pdf/1803.10122.pdf) - World Models

# Motivations and Context

My main side project in 2019 was a reimplementation of 2018's World Models by Ha & Schmidhuber.

This project dominated any spare time I had - usually one to three days per month.  

A plot of my monthly GitHub activity (see more in the section !):

<center>
	<img src="/assets/world-models/commits-month.png">
	<figcaption>Commits per month.  Not all commits are made equal.</figcaption>
  <div></div>
</center>

Total project AWS costs are below. I spent a total of $3,647 on the project (see more in the section !):

|*Table 1 - AWS costs*                     |   Cost [$] |
|:--------------------|-----------:|
| controller          |       1506 |
| vae-and-memory      |        602 |
| sample-experience   |         95 |
| sample-latent-stats |        255 |
| misc                |         25 |
| **compute-total**       |       2485 |
| s3                  |         54 |
| ebs                 |       1108 |
| **storage-total**       |       1162 |
| **total**               |       3648 |


## Why reimplement a paper?

The original idea for reimplementing a paper came from reading an Open AI job advertisement.  Seeing a tangible goal that could put me in the ballpark of a world class machine learning lab, I set out looking for a paper to reimplement.  

I specifically remember reading **high quality implementation**.  This echoed in my mind as I developed the project.

## Why reimplement World Models?

World Models is one of three machine learning papers that were significant for me.

### DQN

The first blew me away without me even reading it.  I have a memory of seeing a YouTube video DQN playing the Atari game Breakout.  Even though I knew nothing of reinforcement learning, the significance of a machine could learn to play a video game from pixels was made clear.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/V1eYniJ0Rnk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<figcaption>DQN playing Atari Breakout</figcaption>
</center>

I had no way of knowing that the algorithm I was watching would be one I implement four times, or that I would teach the mechanics of DQN and it's evolution into Rainbow over twenty times.

### AlphaGo Zero

The second paper is AlphaGo Zero.  The publication of AlphaGo Zero in October 2017 came out after I had taught my course on reinforcement learning twice.

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

The third paper is World Models. World Models is an example of strong technical work presented well.

Ha & Schmidhuber's 2018 paper was accompanied by a blog post that was both interactive and full of moving images. Alongside this outstanding presentation is technical work where supervised and unsupervised learning providing representations to an evolutionary controller.

I had never worked with any of these techniques before reimplementing World Models. The influence of me learning these techniques has been visible in the projects of my students at [Data Science Retreat](https://www.datascienceretreat.com/).  Shout out to Mack (MDN), Samson (VAE) and Stas (World Models + PPO).  Thank you to them for allowing me to improve my understanding in parallel with theirs.

## The promise of learning a model of the world

We operate in a world with spatial and temporal dimensions.  A **world model** is an abstract representation of these dimensions.  In a control problem, a world model can be useful in a number of different ways.

One use of a world model is to **use their low dimensional, internal representations for control**.  We will see that the World Models agent uses it's vision and memory in this way.  The value of having these low dimensional representations is that both prediction and control are easier in low dimensional spaces.

Another use of a world model is to **generate data for training**.  A model that is able to predict the environment observation and reward transition dynamics can be used recurrently to generate rollouts.

**These two uses can be combined together**, where world models are used to generate rollouts in the low dimensional, internal representation spaces.  This is *learning within a dream*.

The value of these different approaches is clear to anyone who (like I have), has spent time building or learning environment models. The sample inefficiency of modern reinforcement learning agents means that an environment models is required for simulation.

I encountered this problem in industry, as an energy data scientist at Tempus Energy.  We had no simulator, and struggled to learn an environment model from limited amounts of customer data.

## The four competences

<center>
	<img src="/assets/world-models/bach-bacteria.jpg">
</center>

<p></p>

I read Daniel C. Dennet's *From Bacteria to Bach and Back* in 2018.  In the book Dennet introduces **four grades of competence** (competence being the ability to act well).

Each of these four grades is a successive application of generate, test & select (more on that in Evolution).  I have found them invaluable for organizing computational control algorithms.

The first grade is **Darwinian**.  This agent has pre-designed and fixed competences.  It doesn't improve within it's lifetime, with global improvement occuring via local selection.  Examples include cells and computational evolutionary algorithms.

The second grade is **Skinnerian**.  This agent has the ability to improve it's behaviour by learning to respond to reinforcement.  Examples include animals and model-free reinforcement learning.

The third grade is **Popperian**.  This agent learns models of the environment, enabling the ability to locally improve via testing behaviours with the learnt model.  Examples include crows, cats, dogs, dolphins, primates and model-based reinforcement learning.

The final grade is **Gregorian**.  This agent builds thinking tools, such as arithmetic, democracy & computers.  Local improvement is possible via higher order control of mental searches, which are systematic explorations.  The only biological example we have is humans - I do not know of a computational method that builds it's own thinking tools.

The optimization algorithm used to find good parameters of the controller (CMA-ES) is a Darwinian learner.  The full World Models agent is Popperian.

## Markov Decision Process

A Markov Decision Process (MDP) is a mathematical framework for decision making.  Commonly the goal of an agent in an MDP is to maximize the expectation of future rewards.  It can be defined as:

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma, H) $$

- set of states $\mathcal{S}$
- set of actions $\mathcal{A}$
- set of rewards $\mathcal{R}$
- state transition function $ P(s' \mid s,a) $
- reward transition function $ R(r \mid s,a,s') $
- distribution over initial states $d_0$
- discount factor $\gamma$
- horizion $H$

It is common to make the distinction between the state $s$ and observation $x$.  The state represents the true state of the environment and has the Markov property.  

The observation is what the agent sees.  The observation is less informative, and often not Markovian.

Because the World Models agent uses the total episode reward as a learning signal, there is no role for a discount rate $\gamma$.

The data collected by an agent interacting with an environment is a sequence of transitions, with a transition being a tuple of observation, action, reward and next state:

$$ \text{transition} = (x, a, r, x') $$

Both the vision and memory components learn only from the first two elements (observation and action).


# The Environment

Our agent interacts with the `car-racing-v0` environment from OpenAI's `gym` library.  I used the same version of `gym` as the paper codebase (`gym==0.9.4`).

## `car-racing-v0` as a Markov Decision Process

We can describe the `car-racing-v0` environment as a Markov Decision Process.  

In the `car-racing-v0` environment, the agents **observation space** is raw image pixels $(96, 96, 3)$ - this is cropped and resized to $(64, 64, 3)$.

The observation has both a spatial $(96, 96, 3)$ and temporal structure, given the sequential nature of sampling transitions from the environment.

<center>
	<img src="/assets/world-models/f1-final.png">
	<figcaption>The raw observation (96, 96, 3) - the resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

In DQN four environment observations are stacked to make the observation more Markovian.  This isn't done in World Models - the observation is a single frame.

The **action space** has three continuous dimensions - `[steering, gas, break]`.  This is a continuous action space - the most challenging for control.

The **reward** function is $-0.1$ for each frame, $+1000 / N$ for each tile visited, where $N$ is the total tiles on track.  This reward function encourages quickly driving forward on the track.

The **horizon** (aka episode length) is set to $1000$ throughout the paper codebase.  Changing this can have interesting effects on agent performance, and I wonder how significant this hyperparameter is on final agent performance.

## Working with `car-racing-v0`

Of particular importance is using `env.viewer.window.dispatch_events()` in the `reset()` and `step()` methods of the environment ([see the GitHub issue here](https://github.com/openai/gym/issues/976)).  If you don't use it you will get corrupt environment observations!

<center>
	<img src="/assets/world-models/corrupt.jpeg">
	<figcaption>If you see this, your environment observation is corrupt!</figcaption>
  <div></div>
</center>

```python
#  do this before env.step(), env.reset()
self.viewer.window.dispatch_events()
```

See the notebook [`worldmodels/notebooks/car_race_consistency.ipynb`](https://github.com/ADGEfficiency/world-models/blob/master/notebooks/car_race_consistency.ipynb) for more infomation about avoiding corrupt `car-racing-v0` observations.

Below the code for sampling environment observations is given in full ([source here](https://github.com/ADGEfficiency/world-models-dev/blob/master/worldmodels/dataset/car_racing.py)):

```python
# worldmodels/dataset/car_racing.py
# worldmodels/dataset/*
```

# The Agent

<center>
	<img src="/assets/world-models/agent.png">
	<figcaption>The World Models agent</figcaption>
  <div></div>
</center>

In a Markov Decision Process the agent is the **learner and decision maker**.  By interacting with the environment, an agent learns to improve it's policy (rules to select actions).  **The World Models agent has three components.**

The first component is **vision**.  The vision component compresses a high dimensional observation of the environment $x$ to a latent space $z$.  This low dimensional representation is used as one of the two inputs to the controller.

The second component is **memory**.  The memory component predicts the next latent state $z'$ from the current latent state $z$.  The agent never uses the predicted next latent state $z'$ (which represents only one step in the future) but instead the hidden state $h$ is used to predict $z'$.  The low dimensional hidden state is the second input to the controller.

The third component is the **controller**.  The controller is a linear function that takes the vision latent state $z$ and the hidden state $h$ and maps to an action.  The controller parameters are found using an evolutionary algorithm.

It is curious that our agent never uses the final output of either the vision or the memory in the controller.  For both components the agent make use of **internal representations** (of space or time respectively) to make decisions with.

Note also that the vision and memory are **unsupervised**.

# Vision

<center>
	<img src="/assets/world-models/vae.png">
	<figcaption>The World Models vision</figcaption>
  <div></div>
</center>

## Why do we need to see?

Much like us, our agent uses vision to understand the current state of the environment.  In the World Models agent, the vision component is separate.

The World Models vision learns a low dimensional representation of the environment observation.  This gives us a definition of vision as **dimensionality reduction**.  It is the process of reducing high dimensional data into a lower dimensional space.

The canonical example is image classification, where an image can be mapped throughout a convolutional neural network to a predicted class (cat or dog). Another would be the flight or fight response, where visual infomation is mapped to a simple binary decision.

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

## What the controller sees

The vision of the World Models agent reduces the environment observation $x$ (96, 96, 3) into a low dimensional representation $z$ (32,) known as the **latent space**.

The latent representation is **hidden**.  It is unobserved - we have no labels for these 32 variables.  The controller uses the latent representation $z$ as one of its inputs.  It never uses $x$ (or it's reconstruction $x'$) directly.

How do we learn this latent representation if we don't have examples?  One technique is a Variational Autoencoder.

## The Variational Autoencoder

A **Variational Autoencoder (VAE)** forms the vision of our agent.

The VAE is a **generative** model that learns the data generating process.  The data generating process is $P(x,z)$ - the joint distribution over our data (the probability of $x$ and $z$ occurring together).

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

Where do generative models fit in the context of other supervised learning methods?

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

### Convolution

<center>
	<img src="/assets/world-models/conv.png">
	<figcaption>2D convolution with a single filter</figcaption>
  <div></div>
</center>

volume to volume operation, layer = set of filters, pooling = reduces size at cost of infomation

At the heart of convolution is the filter (sometimes called a kernel).  These are usually defined as two dimensional, with the third dimension being set to match the number of channels in the image (3 for RGB).

Different kernels are learnt at different layers - shallower layers learn basic features such as edges, with later layers having filters that detect complex compositions of simpler features.

We can think about these kernels operating on tensors of increasing size:
- matrix (3, 3) * kernel (3, 3) -> scalar (1, )
- image (6, 6, 1) * kernel (3, 3, 1) -> image (6, 6, 1)
- image (6, 6, 1) * n kernels (n, 3, 3, 1) -> tensor (6, 6, 1, n)

Important hyperparameters in convolutional neural networks:
- size of filters (typically 3x3)
- number of filters per layer
- padding
- strides

Due to reusing kernels, the convolutional neural network is translation invariant, meaning the features can be detected in different parts of the images.  This is ideal in image classification.  Max-pooling (commonly used to downsample the size of the internal representation) also produces translation invariance.

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

(Remember we sample our latent variables from our encoder, which is a Gaussian approximation). see also bishop 1994

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

This is how the $\mathbf{KLD}$ is implemented in the VAE loss.  Clip at len(latent) * 0.5


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

<center>
	<img src="/assets/world-models/memory.png">
	<figcaption>The World Models memory</figcaption>
  <div></div>
</center>

The memory of the World Models agent is a discriminative model, that models the conditional probability:

$$ P(z' | z, a) $$

The primary role of the memory in the World Models agent is compression of the future into a low dimensional representation $h$.

This low comression of time $h$ is the hidden state of an LSTM.

The LSTM to trained only to predict the next latent vector $z'$, but learns a longer representation of time via hidden state (specifically $h$)

The memory's life is made easier by being able to predict in the low dimensional space learnt by the VAE.

The memory has two components - an LSTM and a Gaussian Mixture head.  Both these together form a **Mixed Density Network** (MDN).  The MDN was introduced in 1994 by Christopher Bishop.  The MDN was originally introduced with fully connected layers connected to a Gaussian Mixture head.

The motivation behind MDN's is being able to combine a neural network (that can represent arbitrary non-linear functions) with a mixture model (that can model arbitrary conditional distributions).

##  Gaussian Mixtures

In the section ! above we saw that if we make the assumption of Gaussian distributed data, we can derive the mean square error loss function from likelihood maximization.  This loss function leads to learning of the conditional average of the target.

Learning the conditional average can be useful, but also has drawbacks.  For multimodal data, taking the average is unlikely to be informative.

A primary motivation behind using a mixture of distributions is that we can approximate **multi-modal** distributions.

Bishop (?) shows that by training a neural network using a least squares loss function, we are able to learn two statistics.  One is the conditional mean, which is our prediction.  The second statistic is the variance, which we can approximate from the residual.  We can use these two statistics to form a Gaussian.

Being able to learn both the mean and variance motivates the paramterization of a mixture model with Gaussian kernels.

A mixture model is a linear combination of kernel functions:

$$ P(y \mid x) = \sum_{mixes} \alpha(x) \cdot \phi(y \mid x) $$

Where $\alpha$ are mixing coefficients, and $\phi$ is a conditional probability density.  Our kernel of choice is the Gaussian, which has a probability density function:

$$ \phi (z' \mid z, a) = \frac{1}{\sqrt{(2 \pi) \sigma(z, a)}} \exp \Bigg[ - \frac{\lVert z' - \mu(z, a) \rVert^{2}}{2 \sigma(z, a)^{2}} \Bigg] $$

The cool thing about Gaussian mixtures is there ability to approximate complex probability densities using Gaussian's with a diagonal covariance matrix.

Probability distribution output by a mixture can (in principle!) be calculated.  The flexibility is similar to a feed forward neural network, and likely has the same distinction between being able approximate versus being able to learn.

In practice, the mixture probabilities are parameterized as $log \pi$, recovering the probabilities by taking the exponential.  These probabilities are priors of the target having been generated by a mixture component.  These are transformed via a softmax:

$$ \pi = \frac {\exp (\pi)}{\sum exp(\pi)} $$

Meaning our mixture satisfies the constraint:

$$ \sum_{mixes} \pi(z, a) = 1 $$

As with the VAE, the memory parameters are found using likelihood maximization.  One interpretation of likelihood maximization is reducing dissimilarity (Goodfellow)

The parameters $\theta$ are found using likelihood maximization.

$$ M(z' \mid z, a) =  \sum_{mixes} \alpha(z, a) \cdot \phi (z'| z, a) $$

$$ \mathbf{LOSS} = - \log M(z' \mid z, a)$$

$$ \mathbf{LOSS} = - \log  \sum_{mixes} \alpha(z, a) \cdot \phi (z'| z, a) $$

In a more general setting, the variances learnt by a Gaussian mixture can be used as a measure of uncertainty.

A mixture model requires statistics (probabilities, means and variances) as input.  In the World Models memory, these statistics are supplied by a long short-term memory (LSTM) network.

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

The gates can be thought of in terms of the methods of a REST API (GET, PUT and DELETE) or the read, update and delete functions in CRUD.

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

Earlier we introduced the Markov Decision Process (MDP), a mathematical framework for decision making, where an agent interacts with an environment.  One task required by agents in an MDP is credit assignment - determining which actions lead to reward.  An agent that understands credit assignment can use this understanding to select good actions.

In reinforcement learning, credit assignment is often learnt by a function that also learns other tasks, such as learning a low dimensional representation of a high dimensional image.  An example of this is the action-value

In DQN, the action-value function learns to both extract features from the observation and map them to an action.

In the World Models agent these tasks are kept separate, with vision responsible for learning a spatial representation and the memory learning a temporal representation.

This separation allows the use of a simple linear controller, compeletly dedicated to learning how to assign credit.

Another benefit of having a simple, low parameter count controller is that it opens up less sample efficient but more general methods for finding the model parameters.

The downside of this is that our vision and memory might use capacity to learn features that are not useful for control, or not learn features that are useful.

## Evolution

Evolutionary learning is the driving force in our universe.  At the heart of evolution is a paradox - that failure at a low level leads to improvement at a high level.  Examples of learning from failure include biology, business, training neural networks and personal development.

Earlier we introduced the four competences, of which evolutionary algorithms sit right at the bottom.  Evolutionary (or Darwinian) learning is inspired by biological evolution, and is characterized by fixed competence (no learning) within an agents lifetime.

Evolution is iterative improvement using a generate, test, select loop:
- in the **generate** step a population is generated, using infomation from previous steps.
- in the **test** step, the population interacts with the environment, and is assigned a single number as a score.
- in the **select** step, members of the current generation are selected (based on their fitness) to be used to generate the next step.

```
for generation
  generate
  test
  select
```

There is so much to learn from this evolutionary process:
- failure at a low level driving improvement at a higher level
- the effectivness of iterative improvement
- the requirement of a dualistic (agent and environment) view

We now have an understanding of the general process of evolutionary learning.  Now let's look at how we do this *in silico*.

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

From a practical standpoint, the most important features of evolutionary methods are:
- general purpose optimization
- poor sample efficiency
- parallelizability

### General purpose optimization

> Randomized search algorithms are regarded to be robust in a rugged search landscape,which can comprise discontinuities, (sharp) ridges, or local optima
> outliers, noise

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

I hadn't worked with evolutionary algorithms before this project.  Due to possessing a simple mind that relies heaivly on empiciral understanding, I often find implementing algorithms a requirement for understanding.

The simplest evolution strategy (ES) is $(1, \lambda)$-ES.  The algorithm keeps only the best performing member from the previous generation ($N_{best} = 1$), and uses an identity covariance matirx.

I implemented $(1, \lambda)$-ES and a wrapper around `pycma` in a separate repo [ADGEfficiency/evolution](https://github.com/ADGEfficiency/evolution) - please refer to the repo for more detail on the algorithms and optimization problems implemented.

Below is the performance of the $(1, \lambda)$-ES algorithm on the ? problem.

<center>
	<img src="/assets/world-models/sphere-simple-solver.gif">
<figcaption></figcaption>
</center>

There are a number of problems with $(1, \lambda)$-ES.  A major one is a fixed covariance matrix - meaning that even after the approximation of the mean is good, the population is still spread with the same variance.

The evolutionary algorithm the World Models agent addresses this problem, by adapting the covariance matix.  This algorithm is called **CMA-ES**.

## CMA-ES

*[Hansen (2016) The CMA Evolution Strategy: A Tutorial](https://arxiv.org/pdf/1604.00772.pdf)*

The Covariance Matrix Adapation Evolutionary Stragety (CMA-ES) is the evolutionary algorithm used by our World Models agent to find parameters of it's linear controller.

A key feature of CMA-ES is the successive estimation of a full covariance matrix.  When combined with an estimation of the mean, these statistics can be used to form a mulitvariate Gaussian.

Unlike the algorithms we have discussed above, CMA-ES approximates a full covariance matix of our parameter space. This means that we model the pairwise dependencies between parameters - how one parameter changes another.

It is worth pointing out that the Gaussian's we parameterized in the vision and memory components have diagonal covariances, which mean each variable is independent of the changes in all the other variables.

Ha suggests that CMA-ES is effective for up to 10k parameters, as the covariance matrix calculation is $O(n^{2})$.

We can introduce CMA-ES in the context of the generate, test and select loop that defines evolutionary learning.

### Generate

The generation step in CMA-ES involves sampling a population from a multivariate Gaussian, parameterized by a mean $\mu$ and covariance matrix $\mathbf{COV}$:

$$ x \sim \mu + \sigma \cdot \mathbf{N} \Big(0, \mathbf{C} \Big) $$

### Test

The test step in CMA-ES involves parallel rollouts of the population parameters in the environment.  In the World Models agent, each parameter is rolled out 16 times, with the results being averaged to give the fitness for each set of parameters.

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

The method above will approximate the covariance matrix from data (in our case, a population of controller parameters).  We can imagine successively selecting only the best set of parameters, and approximating better covarianco matricies.

The update of $C$ is done using the combination of two updates - rank-one and rank-$\mu$.  Combining these update strategies helps to prevent degeneration with small population sizes, and to improve performance on badly scaled / non-separable problems.

### Rank-one update


In the context of our World Models agent, we might estimate the covarianre of our next population $g+1$ using our samples $x$ and taking a reference mean value from that population.

$$ \mathbf{COV}_{g+1} = \frac{1}{N_{best} - 1} \sum_{pop} \Big( x_{g+1} - \mu_{x_{g+1}} \Big) \Big( x_{g+1} - \mu_{x_{g+1}} \Big) $$

Using the mean of the actual sample $\mu_{g+1}$ leads to an estimation of the covariance within the sample. The approach used in a rank-one update instead uses a reference mean value from the previous generation $g$:

$$ \mathbf{COV}_{g+1} = \frac{1}{N_{best}} \sum_{pop} \Big( x_{g+1} - \mu_{x_{g}} \Big) \Big( x_{g+1} - \mu_{x_{g}} \Big) $$

Using the mean of the previous generation $\mu_{g}$ leads to a covariance matrix that estimates the covariance of the sampled step.

The rank-one update introduces infomation of the correlations between generations using the history of how previous populations have evolved - known as the **evolution path**:

$$ p_{g+1} = (1-c_{c})p_{g} + \sqrt{c_{c} (2-c_{c} \mu_{eff})} \frac{\mu_{g+1} - \mu_{g}}{\sigma_{g}} $$

Where $c_{c}$ and $\mu_{eff}$ are hyperparameters.

The evolution path is a sum over all successive steps, but can be evaluated using only a single step - similar to how we can update a value function over a single transition.  The final form of the rank-one update:

$$ \mathbf{COV}_{g+1} = (1-c_{1}) \mathbf{COV}_{g} + c_{1} p_{g+1} p_{g+1}^{T} $$

### Rank-$\mu$ update

With the small population sizes required by CMA-ES, getting a good estimate of the covariance matirx using a rank-one update is challenging.  The rank-$\mu$ update uses a reference mean value that uses information from all previous generations.  This is done by taking an average over all previous estimated covariance matrices:

$$ \mathbf{COV} = \frac{1}{g+1} \sum_{gens} \frac{1}{\sigma^{2}} \mathbf{COV} $$

We can improve on this by using an exponential weights $w$ to give more influence to recent generations.  CMA-ES also includes a learning rate $c_{\mu}$:

$$ \mathbf{COV}_{g+1} = (1-c_{\mu}) \mathbf{COV} + c_{\mu} \sum_{gens} w \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big) \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big)^{T} $$

Note the introduction of $AA^{T}$ TODO

COV(x) = \sum XX^T

### CMA-ES step-size control

The covariance matrix estimation we see above does not explicitly control for scale.  CMA-ES implements an different evolution path ($p_{sigma}$) that is independent of the covariance matrix update seen above, known as **cumulative step length adapation** (CSA).  This helps to prevent premature convergence.

The intuition CSA is that:
- for short evolution paths, steps are cancelling each other out -> decrease the step size
- for long evolution paths, steps are pointing in the same direction -> increase the step size

To determine whether an observed evolution path is short or long, the path length is compared with the expected length under random selection.

Comparing the observed evolution path with a random (i.e. independent) path allows CMA-ES to determine how to update the step size parameter $c_{\sigma}$.

Our evolution path $p_{\sigma}$ is similar to the evolution path $p$ except it is a conjugate evolution path.  After some massaging (see Tutorial), we end up with a step size update:

$$ \sigma_{g+1} = \sigma_{g} \exp \Big( \frac{c_{\sigma}}{d_{\sigma}} \Big(  \frac{\mid\mid p_{\sigma, g} \mid\mid}{ \mathbf{E} \mid\mid \mathbf{N} (0, I)} - 1 \Big) \Big) $$

Where $c_{\sigma}$ is a hyperparameter controlling the backward time horizon, and $d_{\sigma}$ is a damping parameter.

### The final CMA-ES update

The final CMA-ES updates are:

$$ \mu_{g+1} = \frac{1}{n_{best}} \sum_{n_{best}} x_{g} $$

$$ C_{g+1} = (1 - c_{1} - c_{\mu} \sum w) \cdot C_{g} + c_{1} p_{g+1} \cdot p_{g+1} + c_{\mu} \sum_{gens} w \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big) \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big)^{T} $$

Separate control of the mean, covariance and step-size:
- mean update controlled by $c_{m}$
- covariance matrix $\mathbf{C}$ update controlled by $c_{1}$ and $c_{\mu}$
- step size update controlled by damping parameter $d_{sigma}$

## Implementing the controller & CMA-ES

Above we looked at some of the mechanics of CMA-ES.  Luckily I did not need to reimplement CMA-ES!  Instead I used the wonderful [pycma]().

Using `pycma` required only a simple wrapper class around the ask, evaluate and tell API of pycma.

For each generation the rollout of the linear controller parameters are parallelized using Python's `multiprocessing`.

When using `multiprocessing` with both `pycma` and `tensorflow`, care is required to import these packages at the correct place - within the child process.  Doing these imports in `__main__` will cause you trouble.

The original runs 16 rollouts per generation, with the fitness for a population member being the average across the 16 rollouts.  With a population size of 64, this leads to 1024 (TODO) rollouts per generation.

I also experienced a rather painful bug where some episode rollouts would stall.  This would lead to one of the 64 processes not returning, holding up all the other processes.

My solution to this was a band-aid - putting an alarm on AWS to terminate the instance if the CPU% fell below 50% for ? minutes, along with code to restart the experiment from the latest generation saved in `~/world-models-experiments/control/generations/`.

```python
#worldmodels/control
```

# Timeline

Below is a rough outline of the work done on the ! months.

### April 2019

The first commit I have for this project is **April 6th 2019**.  Work achieved in this month
- sampling a random policy
- the VAE model & training script finished
- memory development (LSTM hidden state)

29 - mdn (nan loss, cgant fine tune, using notebook, num mixes)
- linear combination of kernels

### May

Work achieved in this month
- development of memory model & training scripts
- working on understanding evolutionary methods
- `tf.data`

### June

I didn't work on this project in June - I was busy with lots of teaching for Batch 19 at Data Science Retreat.

### July

Work achieved in this month
- development of memory
- first run of the full agent *Agent One* - achieved an average of 500

assets/world/first.png etc

### August 2019

- transfer from `ADGEfficiency/mono` to `ADGEfficiency/world-models-dev`.
- train second VAE with fixed resize

### September 2019

- train second memory

### October = batch 20 starts

Very little work done in October - I was busy with lots of teaching for Batch 20 at Data Science Retreat.

- working on controller training
- move out of TF 2.0 beta

### November

Work achieved this month:
- controller training development - saving parameters, ability to restart, random seeds for environment
- sampling episodes from trained controller
- train **Agent Two** - problem with the VAE not being able to encode images (i.e. off track), memory trains well - gets confused when on the edge of track
- train **Agent Three** - using data sampled from the controller (5000 episodes),
- train **Agent Four** - using data sampled from the controller, 40 epochs on mem

| Agent | Policy | Episodes | VAE epochs | Memory epochs |
|---|---|---|
|one| random | 10,000 | 10 | 20 |
|two| random | 10,000 | 10 | 20 |
|three| controller two | 5,000 | 10 | 20 |
|four| controller three |5,000 | 15 | 40 |
|five| controller three |5,000 | 15 | 80 |

### December

> Know you don’t hit it on the first generation, don’t think you hit it on the second, on the third generation maybe, on the fourth & fifth, thats when we start talking - Linus Torvalds

This was the final month of technical work (finishing on December 19), where Agent Five was trained.  Work achieved this month:
- training Agent Five
- code to visualize the rollouts of the Agent Five controller
- code cleanup & refactors

### January 2020

- blog post writing
- refactors and code cleanup

### February

- draft one done (13 Feb)
- readme cleanup, code cleanup

# Methods

Step by step to reproduce (one round of retraining)

Bash script to dl weights = TODO

See readme for the full methods.

# Final results

This section summarizes the performance of the final agent, along with training curves for the agent components.

## Agent Five performance

> After 150-200 generations (or around 3 days), it should be enough to get around a mean score of ~ 880, which is pretty close to the required score of 900. If you don’t have a lot of money or credits to burn, I recommend you stop if you are satistifed with a score of 850+ (which is around a day of training). Qualitatively, a score of ~ 850-870 is not that much worse compared to our final agent that achieves 900+, and I don’t want to burn your hard-earned money on cloud credits. To get 900+ it might take weeks (who said getting SOTA was easy? :). The final models are saved in log/*.json and you can test and view them the usual way. (http://blog.otoro.net/2018/06/09/world-models-experiments/)

Reproduce paper plots

Training of VAE, training of mem

# Discussion

## Requirement of an iterative training procedure

The most notable difference between this reimplementation and the 2018 paper is the requirement of iterative training.

Section 5 of Ha & Schmidhuber (2018) notes that they were able to train a world model using a random policy, and that more difficult environments would require an iterative training procedure.

The paper codebase implements a random policy by randomly initializing the VAE, memory and controller parameters.  The reimplementation [ctallec/world-models](https://github.com/ctallec/world-models) has two methods for random action sampling - white noise (using the `gym` `env.action_space.sample()` or as a Brownian motion ([see here](https://github.com/ctallec/world-models/blob/master/utils/misc.py)).  The Brownian motion action sampling is the default.

This suggests that slightly more care is needed than relying on a random policy.  An interesting next step would be to look at optimizing the frequency of the iterative training for a given budget of episode sampling.

## Thoughts on Tensorflow 2.0

## Thoughts on `tf.data`

For datasets larger than memory, batches must loaded from disk as needed.  Holding a buffer of batches makes sense to keep GPU utilization high.

One way to achieve this is using `tf.data`.  The API for this library is challenging, and we were required to use the `tf.data` at three different levels of abstraction (a tensor of floats, multiple tensors and a full dataset).

```python

```

Two types of `tfrecord` files were saved and loaded:
- observations and actions for an episode (random or controller policy) - used to train VAE
- VAE latent statistics for an episode - used to train memory

```python

```

Two configurations of `tf.data.Dataset` were used
- VAE trained using a dataset of shuffled observations
- memory trained using a dataset shuffled on the episode level (need to keep the episode sequence structure)

The coverage for our implementation of `tf.data` is fully tested - see [worldmodels/tests/test_tf_records.py']().

Occasionally I get corrupt records - a smwall helper utility is given in [worldmwodels/utils.py]():

It is possible to load the `.tfrecord` files directly from S3.  As neural network training requires multiple passes over the dataset, it makes more sense to pull these down onto the instance and access them locally.

## AWS lessons

I had experience running compute on AVS beefore this project, but not on setting up an entire account from scratch.  The progress was fairly painless, with a reasonable amound of time conficuring the infrastructure I needed.

Setup involved:
- insatnce allowances
- S3 bucket
- IAM to create a user - give S3 access here
- security group with ports open for SSH
- setup.txt

Custom cli commands for what I needed
- change instance type
- change volume size

```bash
```

Didn't use spot at all!

Crashing of controller training

### AWS costs

Compute costs = EC2

Storage costs = EBS + S3

Breakdown of total costs per component (compute costs only):

| component         |   Cost [$] |   Cost [%] |
|:------------------|-----------:|-----------:|
| controller        |    1309.72 |      40.19 |
| vae-and-memory    |     263.04 |       8.07 |
| sample-experience |      56.68 |       1.74 |
| total             |    3258.87 |     100    |

Per component, per month:

| month   |   controller |   vae-and-memory |   sample-experience |   sample-latent-stats |   misc |   compute-total |    s3 |    ebs |   storage-total |   total |
|:--------|-------------:|-----------------:|--------------------:|----------------------:|-------:|----------------:|------:|-------:|----------------:|--------:|
| 1/04/19 |         0    |             0    |                0    |                  0    |   0    |            0    |  0    |   0    |            0    |    0    |
| 1/05/19 |         0    |             0    |                0    |                  0    |   0    |            0    |  0    |   0    |            0    |    0    |
| 1/06/19 |         0    |            90.94 |               29.37 |                  0    |  16.99 |          137.93 |  0    | 153.23 |          153.23 |  291.16 |
| 1/07/19 |         0    |           208.51 |                0.83 |                254.51 |   8.05 |          471.9  |  0    | 304.72 |          304.72 |  776.62 |
| 1/08/19 |         0    |           144.97 |               16.35 |                  0    |   0    |          162.62 | 15.41 | 447.38 |          462.79 |  625.41 |
| 1/09/19 |         0    |            25.25 |                0    |                  0    |   0    |           25.25 | 11.12 | 854.57 |          865.69 |  890.93 |
| 1/10/19 |       104.51 |             0    |                0    |                  0    |   0    |          104.51 | 11.12 | 108.17 |          119.29 |  223.8  |
| 1/11/19 |       673.23 |             0    |               48.33 |                  0    |   0    |          721.56 | 11.42 | 116.86 |          128.29 |  849.84 |
| 1/12/19 |       728.43 |           132.27 |                0.51 |                  0    |   0    |          861.72 |  4.91 | 285.53 |          290.44 | 1152.16 |

One painful mistake occured in September 2019 - leaving a ~ 1TB SSD volume sitting unconnected for a month, leading to a very expensive month!

<center>
	<img src="/assets/world-models/aws.png" width="300%" height="300%">
<figcaption></figcaption>
</center>

## Improvements

### Thoughts on future hyperparameter tweaking

Most of the time I stuck to using the same hyperparameters as in the paper code base.  Hyperparameters I changed:
- batch size to 256 in the VAE training (originally !)
- CMA-ES `s0` set to 0.5
- amount of training data & epochs for the later iterations of VAE & memory training

As I was working on this project, a number of other hyperparameters that could be optimized came to mind:
- VAE loss balancing - in the paper implementation this is done using a `kl_tolerance` parameter of 0.5

```python
kl_loss = tf.reduce_mean(
		tf.maximum(unclipped_kl_loss, self.kl_tolerance * self.latent_dim)
)
```

- improving the random policy sampling
- number of mixtures of in the mixed density network
- number of rollouts per generation

## What did I learn / takeaways

VAE, MDN, evolution

TF 2

Body of work surrounding (blog posts etc).  Same as Silver for Go

# next steps

# References

(same as ToC)

ref dqn, rainbow
