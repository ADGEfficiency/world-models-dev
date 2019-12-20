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

Before we detail the specific contributions of the VAE, it is worth reflecting on what properties we want from our VAE.

The first is high quality reconstruction.  High quality reconstructions mean that the VAE has learnt an efficient encoding of a given sample image $x$.  This leads to the first term in the VAE loss function - a pixel wise reconstruction loss.

The second is the ability to generate new samples.  Generating new samples requires a latent space that is continuous, with samples that are close together in the latent space producing similar images when decoded.

Producing this latent space requires compression.  Traditional autoencoders compress the latent space by constraining it to a fixed length.  The VAE takes it one step further by including second term in the loss function - the Kulback-Lieber divergence between the latent space and a Standard Normal (mean of zero, variance of one).

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
