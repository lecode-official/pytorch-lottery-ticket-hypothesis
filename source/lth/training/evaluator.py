"""Evaluation of trained neural network models."""

import logging
from typing import Union

import tqdm
import torch
import torchmetrics

from lth.datasets import BaseDataset


class Evaluator:
    """Represents a standard evaluation procedure, which evaluates a model on the complete test split of the dataset and reports the accuracy."""

    def __init__(self, device: Union[int, str, torch.device], model: torch.nn.Module, dataset: BaseDataset) -> None:  # pylint: disable=no-member
        """Initializes a new Evaluator instance.

        Args:
            device (Union[int, str, torch.device]): The device on which the validation is to be performed.
            model (torch.nn.Module): The neural network model that is to be evaluated.
            dataset (BaseDataset): The dataset on which the neural network model is to be evaluated.
        """

        # Stores the arguments for later use
        self.device = device
        self.model = model
        self.dataset = dataset

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Makes sure that the model is on the specified device
        self.model.move_to_device(self.device)

        # Creates the loss function
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

    def evaluate(self) -> float:
        """Evaluates the model.

        Returns:
            float: Returns the accuracy of the model.
        """

        # Since we are only evaluating the model, the gradient does not have to be computed
        with torch.no_grad():

            # Puts the model into evaluation mode (this is important for some layers, like dropout and BatchNorm which have different behavior during
            # training and evaluation)
            self.model.eval()

            # Initializes the loss and accuracy metrics
            mean_loss = torchmetrics.MeanMetric().to(self.device)
            accuracy = torchmetrics.Accuracy().to(self.device)

            # Cycles through the whole test split of the dataset and performs the evaluation
            self.logger.info('Evaluating the model...')
            for inputs, targets in tqdm.tqdm(self.dataset.test_split, unit='batch'):

                # Moves the inputs and the targets to the selected device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Performs a forward pass through the neural network
                predictions = self.model(inputs)
                loss = self.loss_function(predictions, targets)  # pylint: disable=not-callable

                # Updates the training metrics
                mean_loss.update(loss)
                accuracy.update(predictions, targets)

            # Computes the accuracy and reports it to the user
            mean_loss = mean_loss.compute().cpu().numpy().item()
            accuracy = accuracy.compute().cpu().numpy().item()
            self.logger.info('Finished validation, validation loss %1.4f, validation accuracy: %1.2f%%', mean_loss, accuracy * 100)

            # Returns the validation loss and validation accuracy
            return mean_loss, accuracy
