"""Contains the baseline command."""

import logging
from argparse import Namespace

from lth.models import create_model
from lth.datasets import create_dataset
from lth.training.trainer import Trainer
from lth.commands.base import BaseCommand
from lth.training.evaluator import Evaluator
from lth.models.hyperparameters import get_defaults
from lth.pruning.magnitude_pruning import LayerWiseMagnitudePruner


class FindTicketCommand(BaseCommand):
    """Represents a command for finding winning lottery tickets."""

    def __init__(self) -> None:
        """Initializes a new FindTicketCommand instance."""

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.trainer = None
        self.is_aborting = False

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        # Determines the hyperparameters (if the user did not specify them as command line parameters, then they default to model and dataset specific
        # values that are known to work well
        learning_rate, batch_size, number_of_epochs = get_defaults(
            command_line_arguments.model,
            command_line_arguments.dataset,
            command_line_arguments.learning_rate,
            command_line_arguments.batch_size,
            command_line_arguments.number_of_epochs
        )

        # Loads the training and the test split of the dataset and creates the model
        dataset = create_dataset(command_line_arguments.dataset, command_line_arguments.dataset_path, batch_size)
        model = create_model(command_line_arguments.model, dataset.sample_shape[:2], dataset.sample_shape[2], dataset.number_of_classes)

        # Logs out the model and dataset that is being trained on
        self.logger.info(
            'Training %s on %s. Learning rate: %f, batch size: %d, number of epochs: %d',
            model.name,
            dataset.name,
            learning_rate,
            batch_size,
            number_of_epochs
        )

        # Creates the trainer, the evaluator, and the pruner for the lottery ticket creation
        trainer = Trainer(model, dataset)
        evaluator = Evaluator(model, dataset)
        pruner = LayerWiseMagnitudePruner(model)

        # Creates the lottery ticket by repeatedly training and pruning the model
        for iteration in range(command_line_arguments.number_of_iterations):

            # Trains and evaluates the model
            self.logger.info('Starting iteration %d...', iteration + 1)
            trainer.train(learning_rate, number_of_epochs)
            evaluator.evaluate()

            # Creates a new pruning, resets the model to its original weights, and applies the pruning mask, in the last iteration this does not need
            # to be performed as model is not trained again
            if iteration + 1 < command_line_arguments.number_of_iterations:
                pruner.create_pruning_masks()
                model.reset()
                pruner.apply_pruning_masks()
            self.logger.info('Finished iteration %d.', iteration + 1)
