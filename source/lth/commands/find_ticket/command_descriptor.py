"""Contains the descriptor for the find-ticket command."""

from argparse import ArgumentParser

from lth.models import get_model_ids
from lth.datasets import get_dataset_ids
from lth.commands.base import BaseCommandDescriptor


class FindTicketCommandDescriptor(BaseCommandDescriptor):
    """Represents the descriptor for the find-ticket command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'find-ticket'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return '''Performs the Lottery Ticket Algorithm with resetting described by Frankle et al. in "The Lottery Ticket Hypothesis: Finding Sparse,
            Trainable Neural Networks". This algorithm repeatedly trains, prunes, and then retrains a neural network model. After each training and
            pruning cycle, the remaining weights of the neural network are reset to their initial initialization. This results in a sparse neural
            network, which is still trainable from scratch.
        '''

    def add_arguments(self, argument_parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            argument_parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        argument_parser.add_argument(
            'model',
            type=str,
            choices=get_model_ids(),
            help='The name of the model for which a lottery ticket is to be found.'
        )
        argument_parser.add_argument(
            'dataset',
            type=str,
            choices=get_dataset_ids(),
            help='The name of the dataset on which the model is to be trained.'
        )
        argument_parser.add_argument(
            'dataset_path',
            type=str,
            help='The path to the dataset. If it does not exist it is automatically downloaded and stored at the specified location.'
        )
        argument_parser.add_argument(
            '-i',
            '--number-of-iterations',
            dest='number_of_iterations',
            type=int,
            default=20,
            help='The number of train-prune-cycles that are to be performed. Defaults to 20.'
        )
        argument_parser.add_argument(
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
        argument_parser.add_argument(
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
        argument_parser.add_argument(
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
