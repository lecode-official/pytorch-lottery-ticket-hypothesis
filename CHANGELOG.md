# Changelog

## v0.2.0

*Unreleased*

- The trained models and their pruning masks are now saved to disk after each iteration of the iterative magnitude pruning algorithm

## v0.1.0

*Released on October 20, 2022*

- Initial release
- Implements the original lottery ticket hypothesis algorithm using magnitude pruning
- Supports the following models:
  - LeNet-300-100
  - LeNet-5
  - Conv-2
  - Conv-4
  - Conv-6
  - VGG19
- Supports the following datasets:
  - MNIST
  - CIFAR-10
- Supports Linux on AMD64
