# The Lottery Ticket Hypothesis in PyTorch

This repository contains a PyTorch implementation of the Lottery Ticket algorithm introduced by Frankle et al. in **_"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"_** [[1]](#1) and enhanced by Zhou et al. in **_"Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask"_** [[4]](#4). The Lottery Ticket algorithm is capable of producing sparse neural networks that can be trained from scratch. Previously it was thought that it was necessary to train highly over-parameterized neural networks to reach satisfactory accuracies. These neural networks could be pruned afterwards but the pruned networks could not be trained from scratch. Frankle et al. managed to get around this limitation by training the pruned neural network on its original initialization. They found out that re-initializing a pruned neural network resulted in an un-trainable model, but using its original initialization produced highly sparse models with accuracies greater than or equal to the original model in less than or equal training time. Based on their findings, Frankle et al. conjecture the Lottery Ticket Hypothesis:

>>>
A randomly-initialized, dense neural network contains a sub-network that is initialized such that – when trained in isolation – it can match the test accuracy of the original network after training for at most the same number of iterations.
>>>

The original algorithm proposed by Frankle et al. works as follows:

1. Randomly initialize a neural network $f(x;\Theta_0)$ (where $\Theta_0 \sim \mathcal{D}_{\Theta}$).
2. Train the neural network for $j$ iterations, arriving at parameters $\Theta_j$.
3. Prune $p\%$ of the parameters in $\Theta_j$, creating a mask $m$.
4. Reset the remaining parameters to their values in $\Theta_0$, creating the winning ticket $f(x;m \odot \Theta_0)$ (where $\odot$ is the Hadamard, or element-wise, product).

This algorithm is usually performed iteratively by repeatedly training the model, pruning a small percentage of the weights (e.g. 20%), resetting the remaining weights, and training the pruned and reset model again. This algorithm can produce winning tickets for small model architectures on small datasets but fails to produce winning tickets for larger model architectures and datasets without lowering the learning rate and using learning rate warmup. In a follow-up paper **_"Stabilizing the Lottery Ticket Hypothesis"_** [[2]](#2), Frankle et al. extend the Lottery Ticket Hypothesis and its algorithm to rewind the weights to an iteration early in the training instead of resetting them to their initial weights, thus stabilizing the algorithm for larger model architectures and datasets.

Morcos et al. show in their paper **_"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"_** [[3]](#3) that winning tickets are not actually overfit to the dataset or optimizer. They generate tickets using one dataset/optimizer and then successfully train them on other datasets/optimizers.

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

## Contributing

If you would like to contribute, there are multiple ways you can help out. If you find a bug or have a feature request, please feel free to open an issue on [GitHub](https://github.com/lecode-official/lottery-ticket-hypothesis/issues). If you want to contribute code, please fork the repository and use a feature branch. Pull requests are always welcome. Before forking, please open an issue where you describe what you want to do. This helps to align your ideas with mine and may prevent you from doing work, that I am already planning on doing. If you have contributed to the project, please add yourself to the [contributors list](CONTRIBUTORS.md) and add all your changes to the [changelog](CHANGELOG.md). To help speed up the merging of your pull request, please comment and document your code extensively and try to emulate the coding style of the project.

## License

The code in this project is licensed under the MIT license. For more information see the [license file](LICENSE).

## References

<a id="1">**[1]**</a> Jonathan Frankle and Michael Carbin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". In: arXiv e-prints, arXiv:1803.03635 (Mar. 2018), arXiv:1803.03635. arXiv: 1803.03635 [cs.LG].

<a id="1">**[2]**</a> Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. "Stabilizing the Lottery Ticket Hypothesis". In: arXiv e-prints, arXiv:1903.01611 (Mar. 2019), arXiv:1903.01611. arXiv: 1903.01611 [cs.LG].

<a id="1">**[3]**</a> Ari Morcos, Haonan Yu, Michela Paganini, and Yuandong Tian. "One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers". In: Advances in Neural Information Processing Systems. Ed. by H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alch ́e-Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc., 2019. URL: https://proceedings.neurips.cc/paper/2019/file/a4613e8d72a61b3b69b32d040f89ad81-Paper.pdf.

<a id="1">**[4]**</a> Hattie Zhou, Janice Lan, Rosanne Liu, and Jason Yosinski. "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask". In: Advances in Neural Information Processing Systems. Ed. by H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alch ́e-Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc., 2019. url: https://proceedings.neurips.cc/paper/2019/file/113d7a76ffceca1bb350bfe145467c6-Paper.pdf.

## Cite this Repository

If you use this software in your research, please cite it like this or use the "Cite this repository" widget in the about section.

```bibtex
@software{Neumann_Lottery_Ticket_Hypothesis_2022,
    author = {Neumann, David},
    license = {MIT},
    month = {10},
    title = {{Lottery Ticket Hypothesis}},
    url = {https://github.com/lecode-official/lottery-ticket-hypothesis},
    version = {0.1.0},
    year = {2022}
}
```

## To-Do's

1. Rename the `find-ticket` command to `find-winning-ticket`
2. Add support for different mask-0 and mask-1 actions
3. Implement the ResNet-18 model
4. Intelligently retain model checkpoint files
5. Extensively log hyperparameters and training statistics
6. The names of the VGG networks seems to be wrong, they should be renamed
7. General clean up, so that the project can be made public
8. Perform extensive experiments on all supported models and datasets and record the results in the read me
9. Add support for macOS on ARM64
10. Add support for plotting training statistics
