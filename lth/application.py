"""Represents the lottery ticket hypothesis application."""

__version__ = '0.1.0'

import os
import sys
import logging
import datetime
import argparse
import functools

import tqdm

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
        self.number_of_iterations = None
        self.number_of_epochs = None
        self.batch_size = None
        self.learning_rate = None

    def run(self):
        """Runs the application. This is the actual entry-point to the application."""

        # Parses the command line arguments of the application
        self.parse_command_line_arguments()

        # Checks which command the user wants to execute and executes it accordingly
        if self.command == 'find-ticket':
            self.find_ticket()
        else:
            self.logger.error('No command specified, exiting.')

    def find_ticket(self):
        """
        Performs the Lottery Ticket Algorithm with resetting described by Frankle et al. in "The Lottery Ticket Hypothesis: Finding Sparse, Trainable
        Neural Networks".
        """

        # Determines the hyperparameters (if the user did not specify them as command line parameters, then they default to model-specific values that
        # are known to work well
        if self.model == 'lenet5':
            learning_rate = self.learning_rate if self.learning_rate is not None else LeNet5.learning_rate
            batch_size = self.batch_size if self.batch_size is not None else LeNet5.batch_size
            number_of_epochs = self.number_of_epochs if self.number_of_epochs is not None else LeNet5.number_of_epochs
        elif self.model == 'lenet-300-100':
            learning_rate = self.learning_rate if self.learning_rate is not None else LeNet_300_100.learning_rate
            batch_size = self.batch_size if self.batch_size is not None else LeNet_300_100.batch_size
            number_of_epochs = self.number_of_epochs if self.number_of_epochs is not None else LeNet_300_100.number_of_epochs
        else:
            raise ValueError('Unknown model: {0}.'.format(self.dataset))

        # Loads the training and the test split of the dataset
        if self.dataset == 'mnist':
            dataset = Mnist(self.dataset_path, batch_size)
        elif self.dataset == 'cifar10':
            dataset = Cifar10(self.dataset_path, batch_size)
        else:
            raise ValueError('Unknown dataset: {0}.'.format(self.dataset))

        # Creates the model
        if self.model == 'lenet5':
            model = LeNet5(dataset.sample_shape[:2], dataset.sample_shape[2], dataset.number_of_classes)
        elif self.model == 'lenet-300-100':
            model = LeNet_300_100(dataset.sample_shape[:2], dataset.sample_shape[2], dataset.number_of_classes)

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
        for _ in range(self.number_of_iterations):
            trainer.train(learning_rate, number_of_epochs)
            evaluator.evaluate()
            pruner.create_pruning_masks()
            model.reset()
            pruner.apply_pruning_masks()

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
            version='Lottery Ticket Hypothesis Experiments {0}'.format(__version__),
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

        # Adds the command line argument that disables progress bars in the output of the application
        argument_parser.add_argument(
            '-P',
            '--disable-progress-bar',
            dest='disable_progress_bar',
            action='store_true',
            help='''
                Disables the progress bar for all actions in the application. This is helpful if this application is used in a script, then the output
                of the progress bar can be really messy and hard to parse.
            '''
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
        Application.add_find_ticket_command(sub_parsers)

        # Parses the arguments
        arguments = argument_parser.parse_args()
        self.command = arguments.command
        if self.command == 'find-ticket':
            self.model = arguments.model
            self.dataset = arguments.dataset
            self.dataset_path = arguments.dataset_path
            self.number_of_iterations = arguments.number_of_iterations
            self.number_of_epochs = arguments.number_of_epochs
            self.batch_size = arguments.batch_size
            self.learning_rate = arguments.learning_rate

        # Disables the rendering of the progress bar across the whole application, it would be very annoying to check for this everytime, luckily,
        # tqdm has a disable flag, which is globally overwritten using functools (the partial function returns a new function, which internally calls
        # the specified function, were the parameters are set automatically)
        if arguments.disable_progress_bar:
            tqdm.tqdm = functools.partial(tqdm.tqdm, disable=True)

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
    def add_find_ticket_command(sub_parsers):
        """
        Adds the find-ticket command, which performs the Lottery Ticket Algorithm with resetting described by Frankle et al. in "The Lottery Ticket
        Hypothesis: Finding Sparse, Trainable Neural Networks".

        Parameters
        ----------
            sub_parsers: Action
                The sub parsers to which the command is to be added.
        """

        find_ticket_command_parser = sub_parsers.add_parser(
            'find-ticket',
            help='''
                Performs the Lottery Ticket Algorithm with resetting described by Frankle et al. in "The Lottery Ticket Hypothesis: Finding Sparse,
                Trainable Neural Networks". This algorithm repeatedly trains, prunes, and then retrains a neural network model. After each training
                and pruning cycle, the remaining weights of the neural network are reset to their initial initialization. This results in a sparse
                neural network, which is still trainable from scratch.
            '''
        )
        find_ticket_command_parser.add_argument(
            'model',
            type=str,
            choices=['lenet5', 'lenet-300-100'],
            help='The name of the model for which a lottery ticket is to be found.'
        )
        find_ticket_command_parser.add_argument(
            'dataset',
            type=str,
            choices=['mnist', 'cifar10'],
            help='The name of the dataset on which the model is to be trained.'
        )
        find_ticket_command_parser.add_argument(
            'dataset_path',
            type=str,
            help='The path to the dataset. If it does not exist it is automatically downloaded and stored at the specified location.'
        )
        find_ticket_command_parser.add_argument(
            '-i',
            '--number-of-iterations',
            dest='number_of_iterations',
            type=int,
            default=20,
            help='The number of train-prune-cycles that are to be performed. Defaults to 20.'
        )
        find_ticket_command_parser.add_argument(
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
        find_ticket_command_parser.add_argument(
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
        find_ticket_command_parser.add_argument(
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
