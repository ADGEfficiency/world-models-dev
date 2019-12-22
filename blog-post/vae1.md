# VISION

A Variational Autoencoder (VAE) forms the vision of the World Models agent.  

### Constructing the VAE

The VAE is a generative model formed of three components - an encoder, a latent space and a decoder.

The encoder is built from convolutional blocks that map from the input image ($x$) (60 x 80 x 3) to statistics (means & variances) of the latent space. 

The primary function of the encoder is to recognize and encode the latent variables.  In the `Car-Racing-v0` case, these variables could be if the car is off the track, or if the track is bending.

These variables are latent as they are hidden - for a given reconstruction we don't know the values of the latent variables.  We don't even need to confirm that they exist!

By constraining the size of the latent space (length 32) the encoder is forced to learn an efficient compression of the image.

This reconstruction of a high dimensional space through a low dimensional bottleneck is the definition of autoencoding (ISIT?).  

The statistics parameterized by the encoder are used to form a distribution over the latent space - formed from indepedent Gaussians.

$z \sim p(z|x)$

Once a latent space has been sampled, the decoder uses deconvolutional blocks to reconstruct the original image $x$ into $x'$.  

Note that we only need a sampled latent space to generate new images.

### Training the VAE

Abbreviations!

Before we detail the specific contributions of the VAE, it is worth reflecting on what properties we want from our VAE.

The first is high quality reconstruction.  High quality reconstructions mean that the VAE has learnt an efficient encoding of a given sample image $x$.  

The VAE is trained using likelihood maximization - maximizing the joint probability of an image $x$ and a latent state $z$

$p(x,z)$

Learning this joint probability is what generative models do.  It can be directly contrasted with the simpler disciminative model, where the model learns a conditional probability such as $P(y|x)$ (probability of $y$ given $x$).

We can rewrite the joint probability of our VAE as:

$p(x,y) = p(x|z)p(z)$

To generate a new image $x$ we can:
- sample latent variables $z \sim p(z)$
- sample an image $x \sim p(x|z)$

Note that this describes only the latent space and encoder - we don't need the encoder to generate new images (only to train).

When we are training the encoder, the goal is to produce a model that learns $p(z|x)$ - a model that is able to infer good values of the latent variables given our observed data.

Bayes Theorem shows us how to decompose the encoder:

$p(z|x) = frac{p(x|z) \cdot p(z)}{p(x)}$

https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

The key challenge at this stage is $p(x)$ - calculating this requires evaluating an exponential time integral.  The VAE sidesteps this by approximating the posterior $p(z|x)$ using Gaussians.  $e(z|x)$ is our encoder.

How well does our encoder approximate the true posterior $p(z|x)$?  We can measure this using the Kullback-Leibler divergence (KLD).  The KLD has a number of interpretations:
- measures the information lost when using q to approximate p

$ KL(q(z|x) || e(z|x)) = E_p[log e(z|x)] - E_p[log p(x, z)] + log p(x)$

Now for another trick from the VAE, which results in replacing computing and minimizing this KLD with Evidence Lower Bound (ELBO) maximization, .  Expanding the notation to show the parameters of the encoder ($\theta$) and the decoder ($\omega$):

$ELBO(\theta, \omega) =  E_{z \sim e_{\theta}(z|x)} [\log d(x|z)] $

The last step is to convert the ELBO maximization into the more familiar loss function minimization, which results in the VAE loss function's final form:

$ L(\theta, \omega) = - above $

Remember that the loss function above is the result of minimizing the KLD between our encoder $e(z|x)$ and the true distribution $p(z|x)$.  What we have is a result of maximizing the log-likelihood of the data.

*gaussian thing*

The final step is writing this in code (autodiff) -> grads




## Understanding the VAE loss function

Now that we have a loss function, we can interpret each of the two terms.


This leads to the first term in the VAE loss function - the negative log likelihood of the decoder, when sampling from a latent space parameterized by the encoder:

$$ 

https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencode://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder

In practice this term is implemented using a pixel wise reconstruction loss.  https://stats.stackexchange.com/questions/288451/why-is-mean-squared-error-the-cross-entropy-between-the-empirical-distribution-a/288453

This can be done because

x | z \sim N(u, sig)

which is normal -> log likelihood of gaussian distribution and L2 loss

https://stats.stackexchange.com/questions/288451/why-is-mean-squared-error-the-cross-entropy-between-the-empirical-distribution-a/288453

Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model. For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model.










- a pixel wise reconstruction loss.

The second is the ability to generate new samples.  Generating new samples requires a latent space that is continuous, with samples that are close together in the latent space producing similar images when decoded.  This requirement is a challenge in traditional autoencoders, which learn spread out latent spaces.

The VAE tackles this problem by making the encoding stochastic.  Because the latent space fed to the decoder is spread (controlled by the parameterized variance of the latent space distribution), it learns to decode a range of variatons for a given $x$.

Thus the latent space being stochastic helps to make it continuous.  This latent space also requires compression.  Traditional autoencoders compress the latent space by constraining it to a fixed length.  The VAE takes it one step further by including second term in the loss function - a Kulback-Lieber divergence (KLD).

The KLD is a measurement of how different two probability distributions are (note I didn't say distance!).  But which two?

The first term is the latent space - intutive as we want to improve the quality of the latent space.

The second term in the VAE KLD is the standard normal distribution (a normal with mean of zero, variance of one).  

Minimizing the KLD means we are trying to make the latent space look like random noise.  It encourages putting encodings near the center of the latent space.  

When combined with the reconstruction loss, the loss function of the VAE is complete.

$ L(\theta, \omega) = - 
)





between the latent space and a 

KL = CE - entropy

The KL loss term further compresses the latent space.  This compression means that using a VAE to generate new images requires only sampling from noise!  This ability to sample without input is the definition of a generative model.

Compute the KL analytically

The reason that we can use noise to generate images is that we pass that noise through the complex function that is the decoder.

### The reparameterization trick

reparameterization (allows stochastic gradients)
- 2.3 2019, 4 in 2016

The VAE is therefore stochastic - the latent space is sampled from a distribution parameterized by the encoder.  This sampling requires a reorganization of the latent space from within the model internals to an input to the model.

The reparameterization trick results in a latent space architecture as follows:

$ z = \sigma (x) \cdot n + \mu (x) $

$ n \sim \mathcal{N}(0, 1) $

After the refactor of the randomness, we can now take a gradient of our loss function and train the VAE.  Remember how the VAE integrates into the larger World Models agent - we never use the reconstruction - we only want the latent space.
