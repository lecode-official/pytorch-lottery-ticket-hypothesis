"""Represents the lottery ticket hypothesis application."""

__version__ = '1.0'

import os
import sys
import logging
import datetime
import argparse

from .datasets.mnist import Mnist
from .datasets.cifar import Cifar10
from .training.trainer import Trainer
from .training.evaluator import Evaluator
from .models.lenet import LeNet_300_100, LeNet5
from .pruning.magnitude_pruning import LayerWiseMagnitudePruner

class Application:
    """Represents the lottery ticket hypothesis application."""

    def __init__(self):
        """Initializes a new Application instance."""

        self.logger = None
        self.command = None
        self.model = None
        self.dataset = None
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
            self.logger.error('No command specified, exiting.')

    def train(self):
        """Trains the LeNet5 model on the MNIST dataset."""

        # Loads the training and the test split of the MNIST dataset
        if self.dataset == 'mnist':
            dataset = Mnist(self.dataset_path, self.batch_size)
        elif self.dataset == 'cifar10':
            dataset = Cifar10(self.dataset_path, self.batch_size)
        else:
            raise ValueError('Unknown dataset: {0}.'.format(self.dataset))

        # Creates the model
        if self.model == 'lenet5':
            model = LeNet5(dataset.sample_shape[:2], dataset.sample_shape[2], dataset.number_of_classes)
        elif self.model == 'lenet-300-100':
            model = LeNet_300_100(dataset.sample_shape[:2], dataset.sample_shape[2], dataset.number_of_classes)
        else:
            raise ValueError('Unknown model: {0}.'.format(self.dataset))

        # Logs out the model and dataset that is being trained on
        self.logger.info('Training %s on %s.', model.name, dataset.name)

        # Trains the model on the training split of the dataset
        trainer = Trainer(model, dataset)
        trainer.train(self.learning_rate, self.number_of_epochs)

        # Evaluates the model on the test split of the dataset
        evaluator = Evaluator(model, dataset)
        evaluator.evaluate()

        # Prunes the network
        pruner = LayerWiseMagnitudePruner(model)
        pruning_masks = pruner.create_pruning_masks()
        pruner.apply_pruning_masks(pruning_masks)

        # Evaluates the pruned model on the test split of the dataset
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

        # Adds the command line argument for the version of the application
        argument_parser.add_argument(
            '-V',
            '--verbosity',
            dest='verbosity',
            type=str,
            choices=['all', 'debug', 'info', 'warning', 'error', 'critical'],
            default='debug',
            help='Sets the verbosity level of the logging. Defaults to "debug".'
        )

        # Adds the command line argument for the path to which the log file is to be written
        argument_parser.add_argument(
            '-l',
            '--logging-path',
            dest='logging_path',
            type=str,
            default=None,
            help='''
                The path to which the logging output is to be written. If this is a path to a file, then the log is written into the specified file.
                If the file exists, it is overwritten. If this is a path to a directory, then a log file with a timestamp is created in that
                directory.
            '''
        )

        # Adds the commands
        sub_parsers = argument_parser.add_subparsers(dest='command')
        Application.add_training_command(sub_parsers)

        # Parses the arguments
        arguments = argument_parser.parse_args()
        self.command = arguments.command
        if self.command == 'train':
            self.model = arguments.model
            self.dataset = arguments.dataset
            self.dataset_path = arguments.dataset_path
            self.number_of_epochs = arguments.number_of_epochs
            self.batch_size = arguments.batch_size
            self.learning_rate = arguments.learning_rate

        # Creates the parent logger for the application
        self.logger = logging.getLogger('lth')
        logging_level_map = {
            'all': logging.NOTSET,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        self.logger.setLevel(logging_level_map[arguments.verbosity])
        logging_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_logging_handler = logging.StreamHandler(sys.stdout)
        console_logging_handler.setLevel(logging_level_map[arguments.verbosity])
        console_logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(console_logging_handler)
        if arguments.logging_path is not None:
            logging_file_path = arguments.logging_path
            if os.path.isdir(logging_file_path):
                file_name = '{0}-lth.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                logging_file_path = os.path.join(logging_file_path, file_name)
            file_logging_handler = logging.FileHandler(logging_file_path)
            file_logging_handler.setLevel(logging_level_map[arguments.verbosity])
            file_logging_handler.setFormatter(logging_formatter)
            self.logger.addHandler(file_logging_handler)

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
            'model',
            type=str,
            choices=['lenet5', 'lenet-300-100'],
            help='The name of the model that is to be trained.'
        )
        train_command_parser.add_argument(
            'dataset',
            type=str,
            choices=['mnist', 'cifar10'],
            help='The name of the dataset on which the model is to be trained.'
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
