"""Represents a module containing datasets from the CIFAR family of datasets."""

import torch
import torchvision

from . import dataset_id
from . import BaseDataset


@dataset_id('cifar10')
class Cifar10(BaseDataset):
    """Represents the CIFAR-10 dataset."""

    def __init__(self, path, batch_size):
        """Initializes a new Cifar10 instance.

        Args:
            path (_type_): The path were the CIFAR-10 dataset is stored. If the dataset could not be found, then it is automatically downloaded to the
                specified location.
            batch_size (_type_): The number of samples that are to be batched together.
        """

        # Stores the arguments
        self.path = path
        self.batch_size = batch_size

        # Exposes some information about the dataset
        self.name = 'CIFAR-10'
        self.sample_shape = (32, 32, 3)
        self.number_of_classes = 10

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
