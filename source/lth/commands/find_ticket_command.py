"""Represents a module, which contains a command for finding winning lottery tickets."""

import logging

from . import BaseCommand
from ..models import create_model
from ..models import get_model_ids
from ..models import hyperparameters
from ..datasets import create_dataset
from ..datasets import get_dataset_ids
from ..training.trainer import Trainer
from ..training.evaluator import Evaluator
from ..pruning.magnitude_pruning import LayerWiseMagnitudePruner

class FindTicketCommand(BaseCommand):
    """Represents a command for finding winning lottery tickets."""

    def __init__(self):
        """Initializes a new FindTicketCommand instance."""

        # Adds the name and the description of the command, which will later be used in the command line help
        self.name = 'find-ticket'
        self.description = '''
            Performs the Lottery Ticket Algorithm with resetting described by Frankle et al. in "The Lottery Ticket Hypothesis: Finding Sparse,
            Trainable Neural Networks". This algorithm repeatedly trains, prunes, and then retrains a neural network model. After each training and
            pruning cycle, the remaining weights of the neural network are reset to their initial initialization. This results in a sparse neural
            network, which is still trainable from scratch.
        '''

        # Creates a logger for the command
        self.logger = logging.getLogger('lth.commands.find_ticket_command.FindTicketCommand')

    def add_arguments(self, parser):
        """
        Adds the command line arguments to the command line argument parser.

        Parameters
        ----------
            parser: argparse.ArgumentParser
                The command line argument parser to which the arguments are to be added.
        """

        parser.add_argument(
            'model',
            type=str,
            choices=get_model_ids(),
            help='The name of the model for which a lottery ticket is to be found.'
        )
        parser.add_argument(
            'dataset',
            type=str,
            choices=get_dataset_ids(),
            help='The name of the dataset on which the model is to be trained.'
        )
        parser.add_argument(
            'dataset_path',
            type=str,
            help='The path to the dataset. If it does not exist it is automatically downloaded and stored at the specified location.'
        )
        parser.add_argument(
            '-i',
            '--number-of-iterations',
            dest='number_of_iterations',
            type=int,
            default=20,
            help='The number of train-prune-cycles that are to be performed. Defaults to 20.'
        )
        parser.add_argument(
            '-e',
            '--number-of-epochs',
            dest='number_of_epochs',
            type=int,
            default=None,
            help='''
                The number of epochs to train the neural network model for. If not specified, then it defaults to a model-specific value that is known
                to work well.
            '''
        )
        parser.add_argument(
            '-b',
            '--batch-size',
            dest='batch_size',
            type=int,
            default=None,
            help='''
                The size of the mini-batch used during training and testing. If not specified, then it defaults to a model-specific value that is
                known to work well.
            '''
        )
        parser.add_argument(
            '-l',
            '--learning-rate',
            dest='learning_rate',
            type=float,
            default=None,
            help='''
                The learning rate used in the training of the model. If not specified, then it defaults to a model-specific value that is known to
                work well.
            '''
        )

    def run(self, command_line_arguments):
        """
        Runs the command.

        Parameters:
        -----------
            command_line_arguments: argparse.Namespace
                The parsed command line arguments.
        """

        # Determines the hyperparameters (if the user did not specify them as command line parameters, then they default to model and dataset specific
        # values that are known to work well
        learning_rate, batch_size, number_of_epochs = hyperparameters.get_defaults(
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
