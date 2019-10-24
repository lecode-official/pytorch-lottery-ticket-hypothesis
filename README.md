# Lottery Ticket Hypothesis

This repository contains a PyTorch implementation of the Lottery Ticket algorithm introduced by Frankle et al. in **_"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"_** and enhanced by Zhou et al. in **_"Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask"_**. The Lottery Ticket algorithm is capable of producing sparse neural networks that can be trained from scratch. Previously it was thought that it was necessary to train highly over-parameterized neural networks to reach satisfactory accuracies. These neural networks could be pruned afterwards but the pruned networks could not be trained from scratch. Frankle et al. managed to get around this limitation by training the pruned neural network on its original initialization. They found out that re-initializing a pruned neural network resulted in an un-trainable model, but using its original initialization produced highly sparse models with accuracies greater than or equal to the original model in less than or equal training time. Based on their findings, Frankle et al. conjecture the Lottery Ticket Hypothesis:

>>>
A randomly-initialized, dense neural network contains a sub-network that is initialized such that -- when trained in isolation -- it can match the test accuracy of the original network after training for at most the same number of iterations.
>>>

The original algorithm proposed by Frankle et al. works as follows:

1. Randomly initialize a neural network $`f(x;\Theta_0)`$ (where $`\Theta_0 \sim \mathcal{D}_{\Theta}`$).
1. Train the neural network for $`j`$ iterations, arriving at parameters $`\Theta_j`$.
1. Prune $`p\%`$ of the parameters in $`\Theta_j`$, creating a mask $`m`$.
1. Reset the remaining parameters to their values in $`\Theta_0`$, creating the winning ticket $`f(x;m \odot \Theta_0)`$ ($`\odot`$ being the Hadamard, or element-wise, product).

This algorithm is usually performed iteratively by repeatedly training the model, pruning a small percentage of the weights (e.g. 20%), resetting the remaining weights, and training the pruned and reset model again. This algorithm can produce winning tickets for small model architectures on small datasets but fails to produce winning tickets for larger model architectures and datasets without lowering the learning rate and using learning rate warmup. In a follow-up paper **_"Stabilizing the Lottery Ticket Hypothesis"_**, Frankle et al. extend the Lottery Ticket Hypothesis and its algorithm to rewind the weights to an iteration early in the training instead of resetting them to their initial weights, thus stabilizing the algorithm for larger model architectures and datasets.

Morcos et al. show in their paper **_"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"_** that winning tickets are not actually overfit to the dataset or optimizer. They generate tickets using one dataset/optimizer and then successfully train them on other datasets/optimizers.

Zhou et al. explore winning tickets in-depth and extended the algorithm to adapt it to different methods of producing masks (mask criterion) and different actions on weights whose mask value is 0 or 1 (mask-0 action and mask-1 action respectively). They use this extension to analyze winning tickets and their creation. Their proposed algorithm works as follows:

1. Initialize a mask $`m`$ to all ones. Randomly initialize the parameters $`\Theta`$ of a network $`f(x;\Theta \odot m)`$.
1. Train the parameters $`w`$ of the network $`f(x;\Theta \odot m)`$ to completion. Denote the initial weights before training $`\Theta_0`$ and the final weights after training for $`j`$ iterations $`\Theta_j`$.
1. *Mask Criterion*. Use the mask criterion $`M(\Theta_0, \Theta_j)`$ to produce a masking score for each currently unmasked weight. Rank the weights in each layer by their scores, set the mask value for the top $`p\%`$ to $`1`$, the bottom $`(100 - p)\%`$ tp $`0`$, breaking ties randomly. Here $`p`$ may vary by layer. $`M(\Theta_0, \Theta_j) = |\Theta_j|`$ is the corresponding mask criterion used by Frankle et al.
1. *Mask-1 Action*. Take some action with the weights with the mask value $`1`$. Frankle et al. reset these weights to their initial value.
1. *Mask-0 Action*. Take some action with the weights with the mask value $`0`$. Frankle et al. pruned these weights, i.e. set them to $`0`$ and froze them during subsequent training.
1. Repeat from 1 if performing iterative pruning.

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

## Interesting Research Avenues and Experiment Ideas

1. Perform the usual Lottery Ticket algorithm but re-initialize pruned weights instead of setting them to 0. If winning tickets are so good, because their initial weights won the initialization lottery, then this procedure would increase the chances of finding good initial values for all weights. In this procedure one would have to start with a larger pruning rate and then decrease it over time. If this procedure works, then the neural network should reach better accuracy than its normally trained counterpart.
1. Instead of performing the usual Lottery Ticket algorithm, a mask could be created after each epoch of training, then the "pruned" weights could be slightly perturbed during the next epoch (selective dropout with a non-Bernoulli distribution). This could be interpreted as changing the initialization of the weight after the fact, slightly nudging it to increase the chance of getting a good initialization of the weight. If this really works, then this could be a novel explanation for why dropout (and stochasticity in general) works so well during training.
1. Frankle et al. only used a very simple heuristic for identifying weights to prune (weights with a low magnitude are pruned: $`|w_f|`$). Zhou et al. try other heuristics, which are very similar to the original idea in that they only use the initial weights and the final weights to identify prunable weights. They find that "magnitude increase" ($`|w_f| - |w_i|`$) generally works slightly better than the original magnitude pruning (which they call "large final"). Also, "movement" ($`|w_f - w_i|`$) and "large init, large final" ($`min(\alpha |w_f|, |w_i|)`$) also tend to work okay (but not as good as "large final" and "movement"). This begs the question if other pruning methods that take more information into account could either produce better winning tickets (higher accuracy) or find winning tickets faster (higher pruning rate: $`p\%`$). One interesting candidate could be LRP pruning (from what I have heard, using ten samples per class enables one to prune 60% of the weights without loss in accuracy).
1. Right now all papers concerning the Lottery Ticket Hypothesis have only tried unstructured pruning. This is very interesting research but has no "real-world impact" as it only shows that sparse/small, trainable neural networks exist. Since they use unstructured pruning, the resulting models may be sparse, but they have no gain in size (unless compressed) and no gain in speed (unless resorting to specialized sparse matrix algebra software and maybe specialized hardware). Structured pruning on the other hand allows one to prune away whole structures (e.g. convolution kernels) from the model thus actually decreasing size and computational complexity. If structured pruning works well with the Lottery Ticket Algorithm, then it could be used to find small trainable neural networks that can actually be reused.
1. Previous research only shows that winning tickets are sufficient for training neural networks efficiently, i.e. it has been shown that there is a winning ticket in a trained model. But it is unclear if winning tickets (that is well initialized sub-networks) are necessary to train neural networks efficiently. It would be interesting to see what would happen if we find a winning ticket using the Lottery Ticket algorithm and then remove it from the original neural network (e.g. by resetting pruned weights to their original initialization, instead of setting them to 0, and re-initializing not pruned values instead of resetting them to their original initialization). If such a network is then trained, will it only reach a lower accuracy than the model that contained the winning ticket? If so, this would mean that a neural network only contains a single winning ticket (which seems unlikely). If not, then how many winning tickets does a neural network contain, i.e. how many times can this procedure of removing a winning ticket be repeated until the accuracy of the resulting model drops?
1. The follow-up paper from Frankle et al. introduces the Lottery Ticket algorithm with rewinding. It seems that the only reason why they do this is to avoid learning rate warmup. This is exactly the problem that is tackled by the novel RAdam optimizer (Rectified Adam). It would be interesting to see if using RAdam winning tickets could be found consistently on large models and datasets. Also it would be interesting, that, if it works, if the winning tickets can be used with other optimizers.
1. In their paper, Morcos et al. only show that winning tickets are transferrable between data domains (e.g. between to image datasets and not between an image and a sound dataset) and tasks (e.g. between to classification tasks and not between classification and regression). It would be interesting to see if winning tickets also generalize across domains and tasks.
1. It is very unlikely that this will work, but it would be interesting to see if two winning tickets could be combined into a single ticket. If is does work, then it would be interesting if the higher capacity of the combined winning ticket allows for higher test accuracy.
