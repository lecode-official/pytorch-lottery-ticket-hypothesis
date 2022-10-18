"""Represents a module, which contains the command line commands of the application."""

import os
import glob
import inspect

class BaseCommand:
    """Represents the base class for all commands in the application."""

def get_commands():
    """
    Retrieves a list of all the commands of the application

    Returns
    -------
        list
            Returns a list containing the classes of all the commands of the application.
    """

    # Gets all the other Python modules that are in the commands module
    command_modules = []
    for module_path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.py')):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        command_modules.append(__import__('lth.commands.{0}'.format(module_name), fromlist=['']))

    # Gets the command classes, which are all the classes in the commands module and its sub-modules that inherit from BaseCommand
    command_classes = []
    for module in command_modules:
        for _, module_class in inspect.getmembers(module, inspect.isclass):
            if BaseCommand in module_class.__bases__ and module_class not in command_classes:
                command_classes.append(module_class)

    # Returns the list of command classes
    return command_classes
