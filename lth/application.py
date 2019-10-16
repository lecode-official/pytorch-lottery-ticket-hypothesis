"""Represents the lottery ticket hypothesis application."""

import argparse

class Application:
    """Represents the lottery ticket hypothesis application."""

    def __init__(self):
        """Initializes a new Application instance."""

        self.argument_parser = argparse.ArgumentParser(
            prog='lth',
            description='Represents a command line interface for experiments concerning the lottery ticket hypothesis.'
        )

        sub_parsers = self.argument_parser.add_subparsers(dest='command')
        hello_world_command = sub_parsers.add_parser('hello-world', help='Greets the world.')
        hello_world_command.add_argument('-n', '--name', dest='name', type=str, help='The name of the person that is to be greeted.')

        self.arguments = self.argument_parser.parse_args()

    def run(self):
        """Runs the application. This is the actual entry-point to the application."""

        if self.arguments.command == 'hello-world':
            self.hello_world()

    def hello_world(self):
        """Greets the world."""

        print('Hello, World!')
        print('Top of the morning to you, {0}.'.format(self.arguments.name))
