"""Training procedures for the neural network models of the application."""

import logging
from typing import Union

import tqdm
import torch

from lth.datasets import BaseDataset


class Trainer:
    """Represents a standard training for training a neural network model."""

    def __init__(
            self,
            device: Union[int, str, torch.device],  # pylint: disable=no-member
            model: torch.nn.Module,
            dataset: BaseDataset,
            learning_rate: float) -> None:
        """Initializes a new Trainer instance.

        Args:
            device (Union[int, str, torch.device]): The device on which the model is to be trained.
            model (torch.nn.Module): The neural network model that is to be trained.
            dataset (BaseDataset): The dataset that is used for the training of the model.
            learning_rate (float): The learning rate that is to be used for the training.
        """

        # Stores the arguments for later use
        self.device = device
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Makes sure that the model is on the specified device
        self.model.move_to_device(self.device)

        # Creates the loss function
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        # Creates the optimizer for the training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, number_of_epochs: int) -> None:
        """Starts the training of the neural network.

        Args:
            number_of_epochs (int): The number of epochs for which the model is to be trained.
        """

        # Puts the model in training mode (this is important for some layers, like dropout and BatchNorm which have different behavior during training
        # and evaluation)
        self.model.train()

        # Trains the neural network for multiple epochs
        self.logger.info('Starting training...')
        for epoch in range(number_of_epochs):

            # Cycles through all batches in the dataset and trains the neural network
            cumulative_loss = 0
            for inputs, targets in tqdm.tqdm(self.dataset.training_split, desc=f'Epoch {epoch + 1}', unit='batch'):

                # Resets the gradients of the optimizer
                self.optimizer.zero_grad()

                # Moves the inputs and the targets to the selected device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Performs a forward pass through the neural network
                predictions = self.model(inputs)
                loss = self.loss_function(predictions, targets)  # pylint: disable=not-callable

                # Computes the gradients and applies the pruning mask to it, this makes sure that all pruned weights are frozen and do not get updated
                loss.backward()
                for layer_name in self.model.get_layer_names():
                    layer = self.model.get_layer(layer_name)
                    layer.weights.grad *= layer.pruning_mask
                self.optimizer.step()
                cumulative_loss += loss.item()

            # Reports the average loss for the epoch
            loss = cumulative_loss / len(self.dataset.training_split)
            self.logger.info('Finished Epoch %d, average loss %f.', epoch + 1, round(loss, 4))

        # Reports that the training has finished
        self.logger.info('Finished the training.')
