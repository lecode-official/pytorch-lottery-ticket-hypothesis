# The Lottery Ticket Hypothesis in PyTorch

This repository contains a PyTorch implementation of the Lottery Ticket algorithm introduced by Frankle et al. in **_"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"_** [[1]](#1) and enhanced by Zhou et al. in **_"Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask"_** [[8]](#8). The Lottery Ticket algorithm is capable of producing sparse neural networks that can be trained from scratch. Previously it was thought that it was necessary to train highly over-parameterized neural networks to reach satisfactory accuracies. These neural networks could be pruned afterwards but the pruned networks could not be trained from scratch. Frankle et al. managed to get around this limitation by training the pruned neural network on its original initialization. They found out that re-initializing a pruned neural network resulted in an un-trainable model, but using its original initialization produced highly sparse models with accuracies greater than or equal to the original model in less than or equal training time. Based on their findings, Frankle et al. conjecture the Lottery Ticket Hypothesis:

>>>
A randomly-initialized, dense neural network contains a sub-network that is initialized such that – when trained in isolation – it can match the test accuracy of the original network after training for at most the same number of iterations.
>>>

The original algorithm proposed by Frankle et al. works as follows [[1]](#1):

1. Randomly initialize a neural network $f(x;\theta_0)$ (where $\theta_0 \sim \mathcal{D}_{\theta}$)
2. Initialize a pruning mask $m \in \{0, 1\}^{|\theta|}$ with all ones
3. Train the neural network $f(x; m \odot \theta)$ for $j$ iterations, arriving at parameters $\theta_j$ (where $\odot$ is the Hadamard, or element-wise, product)
4. Update the pruning mask $m$, by pruning $p\%$ of the parameters in $\theta_j$, e.g., using layer-wise or global magnitude pruning, that have not been pruned before
5. Reset the remaining weights back to their original values in $\theta_0$, i.e., $\theta = \theta_0$
6. Repeat steps 3. through 5. until the desired sparsity is reached
7. Reset the remaining parameters to their values in $\theta_0$, creating the winning ticket $f(x; m \odot \theta_0)$

This algorithm can produce winning tickets for small model architectures on small datasets but fails to produce winning tickets for larger model architectures and datasets without lowering the learning rate and using learning rate warmup. In a follow-up paper **_"Stabilizing the Lottery Ticket Hypothesis"_** [[2]](#2), Frankle et al. extend the Lottery Ticket Hypothesis and its algorithm to rewind the weights to an iteration early in the training instead of resetting them to their initial weights, thus stabilizing the algorithm for larger model architectures and datasets.

Morcos et al. show in their paper **_"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"_** [[6]](#6) that winning tickets are not actually overfit to the dataset or optimizer. They generate tickets using one dataset/optimizer and then successfully train them on other datasets/optimizers.

Zhou et al. explore winning tickets in-depth and extended the algorithm to adapt it to different methods of producing masks (mask criterion) and different actions on weights whose mask value is 0 or 1 (mask-0 action and mask-1 action respectively). They use this extension to analyze winning tickets and their creation. Their proposed algorithm works as follows:

1. Randomly initialize a neural network $f(x;\theta_0)$
2. Initialize a pruning mask $m \in \{0, 1\}^{|\theta|}$ with all ones
3. Train the neural network $f(x; m \odot \theta)$ for $j$ iterations, arriving at parameters $\theta_j$
4. *Mask Criterion*. Use the mask criterion $M(\theta_0, \theta_j)$ to produce a masking score for each currently unmasked weight. Rank the weights in each layer by their scores, set the mask value for the top $p\%$ to $1$ and the bottom $(100 - p)\%$ tp $0$, breaking ties randomly. Here $p$ may vary by layer. In this framework, $M(\theta_0, \theta_j) = |\theta_j|$ is the corresponding mask criterion used by Frankle et al.
5. *Mask-1 Action*. Perform some action on the weights with the mask value $1$. Frankle et al. reset these weights to their initial value
6. *Mask-0 Action*. Perform some action on the weights with the mask value $0$. Frankle et al. pruned these weights, i.e., set them to $0$ and froze them during subsequent training
7. Repeat steps 3. through 6. until the desired sparsity is reached

The code base in this repository should serve as the basis to perform further research into why and how these winning tickets work.

## Getting Started

In order to get the lottery ticket hypothesis package and run it, you need to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and, if necessary, [Git](https://git-scm.com/downloads). After that, you are ready to clone the project:

```bash
git clone https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis.git
cd pytorch-lottery-ticket-hypothesis/source
```

Before running the lottery ticket hypothesis package, all dependencies have to be installed, which can easily be achieved using Miniconda. There are different environment files for the different operating systems and platforms. Please select an environment file that fits your operating system and platform. Currently only Linux AMD64 is  officially supported.

```bash
conda env create -f environment.<operating-system>-<architecture>.yaml
```

To use the virtual environment it must be activated first. After using the environment it has to be deactivated:

```bash
conda activate lth
python -m lth <command> <arguments...>
conda deactivate
```

## Finding Winning Tickets

The project consists of a single Python module called `lth`, which offers commands for performing experiments. Finding a winning ticket for a model and a dataset can be done using the `find-ticket` command. For example, the following command performs 20 iterations of the Lottery Ticket algorithm on LeNet-300-100 and MNIST, where each training step consists of 50 epochs:

```bash
python -m lth find-ticket lenet-300-100 mnist ./datasets
```

Currently the following models and datasets are supported:

**Models:**

- LeNet-300-100 [[4]](#4) (`lenet-300-100`)
- LeNet-5 [[4]](#4) (`lenet-5`)
- VGG5 [[7]](#7) (`vgg5`)
- VGG7 [[7]](#7) (`vgg7`)
- VGG9 [[7]](#7) (`vgg9`)
- VGG17 [[7]](#7) (`vgg17`)

**Datasets:**

- MNIST [[5]](#5) (`mnist`)
- CIFAR-10 [[3]](#3) (`cifar-10`)

Detailed information about all the commands as well as their parameters can be found using the `--help` flag:

```bash
python -m lth --help # For general information about the application as well as a list of available commands
python -m lth <command-name> --help # For information about the specified command
```

## Development

When you install new packages during development, please update the environment file. Please make sure to either create a new environment file for your operating system and platform (i.e., choose a moniker in the format `<operating-system>-<architecture>`, e.g., `windows-amd64`), or overwrite the one that matches your operating system and platform. Ideally, try to update all supported environments if you plan on creating a pull request. The environment file can be updated like so:

```bash
conda env export | grep -v "prefix" > environment.<operating-system>-<architecture>.yaml
```

When someone else has added or removed dependencies from the environment, you have to update your environment from the Anaconda environment file as well. Again, please make sure to select the environment that fits your operating system and platform. The `--prune` switch makes sure that dependencies that have been removed from the Anaconda environment file are uninstalled:

```bash
conda env update --file environment.<operating-system>-<architecture>.yaml --prune
```

The code in this repository follows most of the rules of PyLint, as well as PyCodeStyle (see `.pylintrc` and `.pycodestyle`, which contain the settings for both of them). Before committing any changes, adherence to these rules must be checked as follows:

```bash
pylint lth
pycodestyle --config=.pycodestyle lth
```

## Contributing

If you would like to contribute, there are multiple ways you can help out. If you find a bug or have a feature request, please feel free to open an issue on [GitHub](https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis/issues). If you want to contribute code, please fork the repository and use a feature branch. Pull requests are always welcome. Before forking, please open an issue where you describe what you want to do. This helps to align your ideas with mine and may prevent you from doing work, that I am already planning on doing. If you have contributed to the project, please add yourself to the [contributors list](CONTRIBUTORS.md) and add all your changes to the [changelog](CHANGELOG.md). To help speed up the merging of your pull request, please comment and document your code extensively and try to emulate the coding style of the project.

## License

The code in this project is licensed under the MIT license. For more information see the [license file](LICENSE).

## Cite this Repository

If you use this software in your research, please cite it like this or use the "Cite this repository" widget in the about section.

```bibtex
@software{Neumann_Lottery_Ticket_Hypothesis_2022,
    author = {Neumann, David},
    license = {MIT},
    month = {10},
    title = {{Lottery Ticket Hypothesis}},
    url = {https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis},
    version = {0.1.0},
    year = {2022}
}
```

## References

<a id="1">**[1]**</a> Jonathan Frankle and Michael Carbin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". In: arXiv e-prints, arXiv:1803.03635 (Mar. 2018), arXiv:1803.03635. arXiv: 1803.03635 [cs.LG].

<a id="2">**[2]**</a> Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. "Stabilizing the Lottery Ticket Hypothesis". In: arXiv e-prints, arXiv:1903.01611 (Mar. 2019), arXiv:1903.01611. arXiv: 1903.01611 [cs.LG].

<a id="3">**[3]**</a> Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The CIFAR-10 Dataset. 2014. URL: http://www.cs.toronto.edu/~kriz/cifar.html.

<a id="4">**[4]**</a> Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition". In: Proceedings of the IEEE 86.11 (1998), pp. 2278–2324. DOI: 10.1109/5.726791.

<a id="5">**[5]**</a> Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL: http://yann.lecun.com/exdb/mnist/.

<a id="6">**[6]**</a> Ari Morcos, Haonan Yu, Michela Paganini, and Yuandong Tian. "One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers". In: Advances in Neural Information Processing Systems. Ed. by H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alch ́e-Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc., 2019. URL: https://proceedings.neurips.cc/paper/2019/file/a4613e8d72a61b3b69b32d040f89ad81-Paper.pdf.

<a id="7">**[7]**</a> Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition". In: arXiv e-prints (Sept. 2014). arXiv: 1409.1556 [cs.CV].

<a id="8">**[8]**</a> Hattie Zhou, Janice Lan, Rosanne Liu, and Jason Yosinski. "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask". In: Advances in Neural Information Processing Systems. Ed. by H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alch ́e-Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc., 2019. url: https://proceedings.neurips.cc/paper/2019/file/113d7a76ffceca1bb350bfe145467c6-Paper.pdf.

## To-Do's

1. Change the style of the Python doc strings
2. The names of the VGG networks seems to be wrong, they should be renamed
3. Intelligently retain model checkpoint files
4. Extensively log hyperparameters and training statistics
5. Add support for plotting training statistics
6. Make it possible to gracefully abort the training process
7. Add support for macOS on ARM64
8. General clean up, so that the project can be made public
9.  Add linting and fix all linter warnings
10. Implement the ResNet-18 model
11. Perform extensive experiments on all supported models and datasets and record the results in the read me
12. Make it possible to redo all of the experiments from the original paper
13. Implement the models that were used in the paper
14. Add support for different mask-0 and mask-1 actions
