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

    def get_input_size(self):
        """
        Retrieves the input size of the samples of the dataset.

        Returns
        -------
            tuple

            tuple
                Returns a tuple containing the dimensions of the dataset samples on the x and y-axis.
        """

        return 28, 28

    def get_number_of_channels(self):
        """
        Retrieves the number of channels of the samples of the dataset.

        Returns
        -------
            int
                Returns the number of channels that the samples of the dataset have.
        """

        return 1

    def get_number_of_classes(self):
        """
        Retrives the number of classes that the dataset has.

        Returns
        -------
            int
                Returns the number of classes of the dataset.
        """

        return 10
