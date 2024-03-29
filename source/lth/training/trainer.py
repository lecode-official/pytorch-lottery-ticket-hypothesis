"""Training procedures for the neural network models of the application."""

import logging

import tqdm
import torch

from lth.datasets import BaseDataset


class Trainer:
    """Represents a standard training for training a neural network model."""

    def __init__(self, model: torch.nn.Module, dataset: BaseDataset) -> None:
        """Initializes a new Trainer instance.

        Args:
            model (torch.nn.Module): The neural network model that is to be trained.
            dataset (BaseDataset): The dataset that is used for the training of the model.
        """

        self.model = model
        self.dataset = dataset
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def train(self, learning_rate: float, number_of_epochs: int) -> None:
        """Starts the training of the neural network.

        Args:
            learning_rate (float): The learning rate that is to be used for the training.
            number_of_epochs (int): The number of epochs for which the model is to be trained.
        """

        # Checks if CUDA is available, in that case the training is performed on the first GPU on the system, otherwise the CPU is used
        if torch.cuda.is_available():
            device = torch.device('cuda:0')  # pylint: disable=no-member
            self.logger.info('Running on the GPU (%s).', torch.cuda.get_device_name(device=device))
        else:
            device = torch.device('cpu')  # pylint: disable=no-member
            self.logger.info('Running on the CPU.')

        # Transfers the model to the selected device
        self.model.move_to_device(device)

        # Puts the model in training mode (this is important for some layers, like dropout and BatchNorm which have different behavior during training
        # and evaluation)
        self.model.train()

        # Defines the loss function and the optimizer for the model
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Trains the neural network for multiple epochs
        self.logger.info('Starting training...')
        for epoch in range(number_of_epochs):

            # Cycles through all batches in the dataset and trains the neural network
            cumulative_loss = 0
            for batch in tqdm.tqdm(self.dataset.training_split, desc=f'Epoch {epoch + 1}', unit='batch'):

                # Gets the current training batch
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Resets the gradients of the optimizer
                optimizer.zero_grad()

                # Performs a forward pass through the neural network
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)

                # Computes the gradients and applies the pruning mask to it, this makes sure that all pruned weights are frozen and do not get updated
                loss.backward()
                for layer_name in self.model.get_layer_names():
                    layer = self.model.get_layer(layer_name)
                    layer.weights.grad *= layer.pruning_mask
                optimizer.step()
                cumulative_loss += loss.item()

            # Reports the average loss for the epoch
            loss = cumulative_loss / len(self.dataset.training_split)
            self.logger.info('Finished Epoch %d, average loss %f.', epoch + 1, round(loss, 4))

        # Reports that the training has finished
        self.logger.info('Finished the training.')
