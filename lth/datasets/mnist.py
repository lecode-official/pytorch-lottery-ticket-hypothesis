"""Represents a module containing MNIST or MNIST-style datasets."""

import torch
import torchvision

class Mnist:
    """Represents the classical MNIST dataset."""

    def __init__(self, path, batch_size):
        """
        Initializes a new Mnist instance.

        Parameters
        ----------
            path: str
                The path where the MNIST dataset is stored. If it does not exist, it is automatically downloaded to the specified location.
            batch_size: int
                The number of samples that are to be batched together.
        """

        # Stores the arguments
        self.path = path
        self.batch_size = batch_size

        # Loads the training split of the dataset
        dataset_training_split = torchvision.datasets.MNIST(root=self.path, train=True, download=True, transform=torchvision.transforms.ToTensor())
        self.training_split = torch.utils.data.DataLoader(dataset_training_split, batch_size=self.batch_size, shuffle=True, num_workers=10)

        # Loads the test split of the dataset
        dataset_test_split = torchvision.datasets.MNIST(root=self.path, train=False, download=True, transform=torchvision.transforms.ToTensor())
        self.test_split = torch.utils.data.DataLoader(dataset_test_split, batch_size=self.batch_size, shuffle=True, num_workers=10)