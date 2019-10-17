"""Represents a module containing datasets from the Cifar family of datasets."""

import torch
import torchvision

class Cifar10:
    """Represents the Cifar10 dataset."""

    def __init__(self, path, batch_size):
        """
        Initializes a new Cifar10 instance.

        Parameters
        ----------
            path: str
                The path were the Cifar10 dataset is stored. If the dataset could not be found, then it is automatically downloaded to the specified
                location.
            batch_size: int
                The number of samples that are to be batched together.
        """

        # Stores the arguments
        self.path = path
        self.batch_size = batch_size

        # Creates the transformation pipeline for the data augmentation and pre-processing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Loads the training split of the dataset
        dataset_training_split = torchvision.datasets.CIFAR10(root=self.path, train=True, download=True, transform=transform)
        self.training_split = torch.utils.data.DataLoader(dataset_training_split, batch_size=self.batch_size, shuffle=True, num_workers=10)

        # Loads the test split of the dataset
        dataset_test_split = torchvision.datasets.CIFAR10(root=self.path, train=False, download=True, transform=transform)
        self.test_split = torch.utils.data.DataLoader(dataset_test_split, batch_size=self.batch_size, shuffle=False, num_workers=10)

    def get_input_size(self):
        """
        Retrieves the input size of the samples of the dataset.

        Returns
        -------
            tuple
                Returns a tuple containing the dimensions of the dataset samples on the x and y-axis.
        """

        return 32, 32

    def get_number_of_channels(self):
        """
        Retrieves the number of channels of the samples of the dataset.

        Returns
        -------
            int
                Returns the number of channels that the samples of the dataset have.
        """

        return 3

    def get_number_of_classes(self):
        """
        Retrives the number of classes that the dataset has.

        Returns
        -------
            int
                Returns the number of classes of the dataset.
        """

        return 10