# Lottery Ticket Hypothesis

This repository contains a PyTorch implementation of the Lottery Ticket algorithm introduced by Frankle et al. in **_"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"_** and enhanced by Zhou et al. in **_"Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask"_**. The Lottery Ticket algorithm is capable of producing sparse neural networks that can be trained from scratch. Previously it was thought that it was necessary to train highly over-parameterized neural networks to reach satisfactory accuracies. These neural networks could be pruned afterwards but the pruned networks could not be trained from scratch. Frankle et al. managed to get around this limitation by training the pruned neural network on its original initialization. They found out that re-initializing a pruned neural network resulted in an un-trainable model, but using its original initialization produced highly sparse models with accuracies greater than or equal to the original model in less than or equal training time. Based on their findings, Frankle et al. conjecture the Lottery Ticket Hypothesis:

>>>
A randomly-initialized, dense neural network contains a sub-network that is initialized such that – when trained in isolation – it can match the test accuracy of the original network after training for at most the same number of iterations.
>>>

The original algorithm proposed by Frankle et al. works as follows:

1. Randomly initialize a neural network $f(x;\Theta_0)$ (where $\Theta_0 \sim \mathcal{D}_{\Theta}$).
2. Train the neural network for $j$ iterations, arriving at parameters $\Theta_j$.
3. Prune $p\%$ of the parameters in $\Theta_j$, creating a mask $m$.
4. Reset the remaining parameters to their values in $\Theta_0$, creating the winning ticket $f(x;m \odot \Theta_0)$ (where $\odot$ is the Hadamard, or element-wise, product).

This algorithm is usually performed iteratively by repeatedly training the model, pruning a small percentage of the weights (e.g. 20%), resetting the remaining weights, and training the pruned and reset model again. This algorithm can produce winning tickets for small model architectures on small datasets but fails to produce winning tickets for larger model architectures and datasets without lowering the learning rate and using learning rate warmup. In a follow-up paper **_"Stabilizing the Lottery Ticket Hypothesis"_**, Frankle et al. extend the Lottery Ticket Hypothesis and its algorithm to rewind the weights to an iteration early in the training instead of resetting them to their initial weights, thus stabilizing the algorithm for larger model architectures and datasets.

Morcos et al. show in their paper **_"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"_** that winning tickets are not actually overfit to the dataset or optimizer. They generate tickets using one dataset/optimizer and then successfully train them on other datasets/optimizers.

Zhou et al. explore winning tickets in-depth and extended the algorithm to adapt it to different methods of producing masks (mask criterion) and different actions on weights whose mask value is 0 or 1 (mask-0 action and mask-1 action respectively). They use this extension to analyze winning tickets and their creation. Their proposed algorithm works as follows:

1. Initialize a mask $m$ to all ones. Randomly initialize the parameters $\Theta$ of a network $f(x;\Theta \odot m)$.
2. Train the parameters $w$ of the network $f(x;\Theta \odot m)$ to completion. Denote the initial weights before training $\Theta_0$ and the final weights after training for $j$ iterations $\Theta_j$.
3. *Mask Criterion*. Use the mask criterion $M(\Theta_0, \Theta_j)$ to produce a masking score for each currently unmasked weight. Rank the weights in each layer by their scores, set the mask value for the top $p\%$ to $1$, the bottom $(100 - p)\%$ tp $0$, breaking ties randomly. Here $p$ may vary by layer. $M(\Theta_0, \Theta_j) = |\Theta_j|$ is the corresponding mask criterion used by Frankle et al.
4. *Mask-1 Action*. Take some action with the weights with the mask value $1$. Frankle et al. reset these weights to their initial value.
5. *Mask-0 Action*. Take some action with the weights with the mask value $0$. Frankle et al. pruned these weights, i.e. set them to $0$ and froze them during subsequent training.
6. Repeat from 1 if performing iterative pruning.

The code base in this repository should serve as the basis to perform further research into why and how these winning tickets work.

## Usage

The project has several dependencies, e.g. PyTorch, that can conveniently be installed using Anaconda. The repository contains a ready-made Anaconda environment which can be installed like this:

```bash
conda env create -f environment.yml
```

Before using the project, the environment has to be activated:

```bash
conda activate lottery-ticket-hypothesis
# Use the lth Python module
conda deactivate # After finishing, the original environment can be restored
```

The project consists of a single Python module called `lth`, which offers commands for performing experiments. Finding a winning ticket for a model and a dataset can be done using the `find-ticket` command. For example, the following command performs 20 iterations of the Lottery Ticket algorithm on LeNet-300-100 and MNIST, where each training step consists of 50 epochs:

```bash
python -m lth find-ticket lenet-300-100 mnist ./datasets
```

Detailed information about all the commands as well as their parameters can be found using the `--help` flag:

```bash
python -m lth --help # For general information about the application as well as a list of available commands
python -m lth <command-name> --help # For information about the specified command
```

## To-Do's

1. Add support for different mask-0 and mask-1 actions.
2. Add support for different optimizers (especially RAdam).
3. Implement the ResNet-18 model.
4. All results of the algorithm should be written to files, so that they can later be evaluated.
5. The names of the VGG networks seems to be wrong, they should be renamed
6. General clean up, so that the project can be made public
