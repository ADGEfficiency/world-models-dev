Berlin photos, post onto medium, mentorcruise
tsne! - map to 2d space!
show final code in all the sections

code snippets everywhere!

put key / important references everywhere

pre knowledge - conditional, marginal distributions, kld, 

scope!
- baselines
- doom

enjoyed working with images

images & examples for everything!

ecosystem
- original paper

Dimensionality reduction is so often the primary function of machine learning that it is a useful working definition of the entire field.

visalization of rollouts
- random
- halfway
- final

## Table of contents

## Motivations

Like the paper, love the blog post

Learn inside a dream

Open AI challenge

Students (mack, samson, stas)

##  The WM agent

The three components

Compression, prediction, control

Compression
- why = decisions (business decisions)
- image to z
- future to h

### VAE

- draft
- images

### MEM, CONTROL

## Timeline

### April

VAE (loss balancing, error on resize), tensorflow beta, tests

6 - start

11 - latent stats sampling

lstm hidden state (tf2 beta)

29 - mdn (nan loss, cgant fine tune, using notebook, num mixes)
- linear combination of kernels

### May

1 memory trainig working

13 evolution stuff!

tf.data

preloads, hard to use

### June

### July

Agent one 

7 run on memory

31 first run on controller (500)

### November

Agent two
- resize
- cant reconstruct (exploration)
- vae poor
- memory good?

22 Agent three
- sample from controller (5000 images)
- vae train well
- memory poor

? Agent four
- vae train well
- memory still poor (20 or 40 epoch on 5000)

### December

Agent five (linus quote)
- vae well
- memory still poor (40 or 80 epoch on 5000)

| agent | VAE epochs | VAE images
|---|---|---|
|four| 15 | 5,000

## Final results

> After 150-200 generations (or around 3 days), it should be enough to get around a mean score of ~ 880, which is pretty close to the required score of 900. If you don’t have a lot of money or credits to burn, I recommend you stop if you are satistifed with a score of 850+ (which is around a day of training). Qualitatively, a score of ~ 850-870 is not that much worse compared to our final agent that achieves 900+, and I don’t want to burn your hard-earned money on cloud credits. To get 900+ it might take weeks (who said getting SOTA was easy? :). The final models are saved in log/*.json and you can test and view them the usual way. (http://blog.otoro.net/2018/06/09/world-models-experiments/)

Reproduce paper plots

## Methods

Each bash command

## AWS lessons 

S3, AMI, security groups

## AWS costs

Per component, per month

## Unfinished

DOOM, dream

## Improvements

VAE loss balancing

Improve random policy sampling (brownian as per pytorch reimpl)

Num mixes?

CMA-ES sigma decay

Really need 16 generations per population?

## What did I learn

AWS

VAE, MDN, evolution