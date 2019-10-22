# Lottery Ticket Hypothesis

This repository contains a PyTorch implementation of the Lottery Ticket algorithm introduced by Frankle et al. in "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" and enhanced by Zhou et al. in "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask". The Lottery Ticket algorithm is capable of producing sparse neural networks that can be trained from scratch. Previously it was thought that it was necessary to train highly over-parameterized neural networks to reach satisfactory accuracies. These neural networks could be pruned afterwards but the pruned networks could not be trained from scratch. Frankle et al. managed to get around this limitation by training the pruned neural network on its original initialization. They found out that re-initializing a pruned neural network resulted in an un-trainable model, but using its original initialization produced highly sparse models with accuracies greater than or equal to the original model in less than or equal training time. Based on their findings, Frankle et al. conjecture the Lottery Ticket Hypothesis:

>>>
A randomly-initialized, dense neural network contains a sub-network that is initialized such that -- when trained in isolation -- it can match the test accuracy of the original network after training for at most the same number of iterations.
>>>

The original algorithm proposed by Frankle et al. works as follows:

1. Randomly initialize a neural network $`f(x;\Theta_0)`$ (where $`\Theta_0 \sim \mathcal{D}_{\Theta}`$).
1. Train the neural network for $`j`$ iterations, arriving at parameters $`\Theta_j`$.
1. Prune $`p\%`$ of the parameters in $`\Theta_j`$, creating a mask $`m`$.
1. Reset the remaining parameters to their values in $`\Theta_0`$, creating the winning ticket $`f(x;m \odot \Theta_0)`$ ($`\odot`$ being the Hadamard, or element-wise, product).

The code base in this repository should serve as the basis to perform further research into why and how these winning tickets work.
