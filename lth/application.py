"""Represents the lottery ticket hypothesis application."""

__version__ = '0.1.0'

import os
import sys
import logging
import datetime
import argparse
import functools

import tqdm

from .commands import get_commands

class Application:
    """Represents the lottery ticket hypothesis application."""

    def __init__(self):
        """Initializes a new Application instance."""

        self.logger = None
        self.commands = None

    def run(self):
        """Runs the application. This is the actual entry-point to the application."""

        # Parses the command line arguments of the application
        arguments = self.parse_command_line_arguments()

        # Finds the command that is to be run
        for command in self.commands:
            if command.name == arguments.command:
                command.run(arguments)

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
        self.commands = []
        command_classes = get_commands()
        sub_parsers = argument_parser.add_subparsers(dest='command')
        for command_class in command_classes:
            command = command_class()
            command_parser = sub_parsers.add_parser(command.name, help=command.description)
            command.add_arguments(command_parser)
            self.commands.append(command)

        # Parses the arguments
        arguments = argument_parser.parse_args()

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

        # Returns the parsed command line arguments
        return arguments
