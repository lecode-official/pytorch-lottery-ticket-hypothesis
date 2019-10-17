"""Represents the lottery ticket hypothesis application."""

__version__ = '1.0'

import argparse

from .models.lenet import LeNet5
from .datasets.mnist import Mnist
from .training.trainer import Trainer
from .training.evaluator import Evaluator

class Application:
    """Represents the lottery ticket hypothesis application."""

    def __init__(self):
        """Initializes a new Application instance."""

        self.command = None
        self.dataset_path = None
        self.number_of_epochs = None
        self.batch_size = None
        self.learning_rate = None

    def run(self):
        """Runs the application. This is the actual entry-point to the application."""

        # Parses the command line arguments of the application
        self.parse_command_line_arguments()

        # Checks which command the user wants to execute and executes it accordingly
        if self.command == 'train':
            self.train()
        else:
            raise ValueError('Unknown command: "{0}".'.format(self.command))

    def train(self):
        """Trains the LeNet5 model on the MNIST dataset."""

        # Loads the training and the test split of the MNIST dataset
        dataset = Mnist(self.dataset_path, self.batch_size)

        # Loads the LeNet5 model and trains it
        lenet5 = LeNet5(input_size=(28, 28))
        trainer = Trainer(lenet5, dataset)
        trainer.train(self.learning_rate, self.number_of_epochs)

        # Evaluates the model on the test split of the dataset
        evaluator = Evaluator(lenet5, dataset)
        evaluator.evaluate()

    def parse_command_line_arguments(self):
        """Parses the command line arguments of the application."""

        # Creates a command line argument parser for the application
        argument_parser = argparse.ArgumentParser(
            prog='lth',
            description='A command line tool for experiments on the Lottery Ticket Hypothesis.',
            add_help=False
        )

        # Adds the command line argument that displays the help message
        argument_parser.add_argument(
            '-h',
            '--help',
            action='help',
            help='Shows this help message and exits.'
        )

        # Adds the command line argument for the version of the application
        argument_parser.add_argument(
            '-v',
            '--version',
            action='version',
            version='PyTorch LeNet5 Training Command Line Interface {0}'.format(__version__),
            help='Displays the version string of the application and exits.'
        )

        # Adds the commands
        sub_parsers = argument_parser.add_subparsers(dest='command')
        Application.add_training_command(sub_parsers)

        # Parses the arguments
        arguments = argument_parser.parse_args()
        self.command = arguments.command
        if self.command == 'train':
            self.dataset_path = arguments.dataset_path
            self.number_of_epochs = arguments.number_of_epochs
            self.batch_size = arguments.batch_size
            self.learning_rate = arguments.learning_rate

    @staticmethod
    def add_training_command(sub_parsers):
        """
        Adds the training command, which trains a neural network.

        Parameters
        ----------
            sub_parsers: Action
                The sub parsers to which the command is to be added.
        """

        train_command_parser = sub_parsers.add_parser(
            'train',
            help='Trains the LeNet5 neural network on the MNIST dataset.'
        )
        train_command_parser.add_argument(
            'dataset_path',
            type=str,
            help='The path to the MNIST dataset. If it does not exist it is automatically downloaded and stored at the specified location.'
        )
        train_command_parser.add_argument(
            '-e',
            '--number-of-epochs',
            dest='number_of_epochs',
            type=int,
            default=10,
            help='The number of epochs to train for. Defaults to 10.'
        )
        train_command_parser.add_argument(
            '-b',
            '--batch-size',
            dest='batch_size',
            type=int,
            default=32,
            help='The size of the mini-batch used during training and testing. Defaults to 32.'
        )
        train_command_parser.add_argument(
            '-l',
            '--learning-rate',
            dest='learning_rate',
            type=float,
            default=0.001,
            help='The learning rate used in the training of the model. Defaults to 0.001.'
        )
