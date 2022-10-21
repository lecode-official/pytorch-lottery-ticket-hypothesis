"""Contains the find-ticket command."""

import os
import csv
import copy
import logging
from datetime import datetime
from argparse import Namespace

import yaml
import torch

from lth.datasets import create_dataset
from lth.commands.base import BaseCommand
from lth.training import Trainer, Evaluator
from lth.pruning import LayerWiseMagnitudePruner
from lth.models import hyperparameters, create_model


class FindTicketCommand(BaseCommand):
    """Represents the find-ticket command, which uses the iterative magnitude pruning algorithm to find winning lottery tickets."""

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

        # Makes sure that the output directory exists
        os.makedirs(command_line_arguments.output_path, exist_ok=True)

        # Determines the hyperparameters (if the user did not specify them as command line parameters, then they default to model and dataset specific
        # values that are known to work well
        model_id = command_line_arguments.model
        dataset_id = command_line_arguments.dataset
        learning_rate, batch_size, number_of_epochs = hyperparameters.get_defaults(
            model_id,
            dataset_id,
            command_line_arguments.learning_rate,
            command_line_arguments.batch_size,
            command_line_arguments.number_of_epochs
        )

        # Prepares the training statistics CSV file by writing the header to file
        with open(os.path.join(command_line_arguments.output_path, 'training-statistics.csv'), 'w', encoding='utf-8') as training_statistics_file:
            csv_writer = csv.writer(training_statistics_file)
            csv_writer.writerow([
                'timestamp',
                'iteration',
                'training_loss',
                'training_accuracy',
                'validation_loss',
                'validation_accuracy',
                'sparsity'
            ])

        # Saves the hyperparameters for later reference
        with open(os.path.join(command_line_arguments.output_path, 'hyperparameters.yaml'), 'w', encoding='utf-8') as hyperparameters_file:
            yaml.dump({
                'model_id': model_id,
                'dataset_id': dataset_id,
                'dataset_path': command_line_arguments.dataset_path,
                'output_path': command_line_arguments.output_path,
                'number_of_iterations': command_line_arguments.number_of_iterations,
                'number_of_epochs': number_of_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }, hyperparameters_file)

        # Checks if CUDA is available, in that case the training is performed on the first GPU on the system, otherwise the CPU is used
        device = 'cpu'
        device_name = 'CPU'
        if torch.cuda.is_available():
            device = 'cuda'
            device_name = torch.cuda.get_device_name(device)
        self.logger.info('Selected %s to perform training...', device_name)

        # Loads the training and the test split of the dataset and creates the model
        dataset = create_dataset(dataset_id, command_line_arguments.dataset_path, batch_size)
        model = create_model(model_id, dataset.sample_shape[:2], dataset.sample_shape[2], dataset.number_of_classes)

        # Logs out the model and dataset that is being trained on
        self.logger.info(
            'Training %s on %s. Learning rate: %f, batch size: %d, number of epochs: %d',
            model.name,
            dataset.name,
            learning_rate,
            batch_size,
            number_of_epochs
        )

        # Creates the evaluator for the model
        evaluator = Evaluator(device, model, dataset)

        # Creates the pruner for the lottery ticket creation
        pruner = LayerWiseMagnitudePruner(model)

        # Creates the lottery ticket by repeatedly training and pruning the model
        for iteration in range(1, command_line_arguments.number_of_iterations + 1):

            # Trains and evaluates the model
            self.logger.info('Starting iteration %d...', iteration)
            trainer = Trainer(device, model, dataset, learning_rate)
            training_loss, training_accuracy = trainer.train(number_of_epochs)
            validation_loss, validation_accuracy = evaluator.evaluate()

            # Creates a new pruning mask
            pruner.create_pruning_masks()

            # Copies the model, so that the trained model can be saved later on
            trained_model_state_dict = copy.deepcopy(model.state_dict())

            # Resets the model to its original initialization and applies the pruning mask
            model.reset()
            sparsity = pruner.apply_pruning_masks()

            # Writes the training statistics into a CSV file
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(command_line_arguments.output_path, 'training-statistics.csv'), 'a', encoding='utf-8') as training_statistics_file:
                csv_writer = csv.writer(training_statistics_file)
                csv_writer.writerow([timestamp, iteration, training_loss, training_accuracy, validation_loss, validation_accuracy, sparsity])

            # Saves the trained model, the lottery ticket, the pruning mask, and the original initialization to disk
            current_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            output_file_path = os.path.join(
                command_line_arguments.output_path,
                f'{current_date_time}-{model_id}-{dataset_id}-{iteration}-iteration-{sparsity:.2f}-sparsity-{validation_accuracy:.2f}-accuracy.pt'
            )
            original_initialization = {}
            pruning_mask = {}
            for layer in model.layers:
                original_initialization[f'{layer.name}.weight'] = layer.initial_weights
                original_initialization[f'{layer.name}.bias'] = layer.initial_biases
                pruning_mask[layer.name] = layer.pruning_mask
            torch.save({
                'trained_model': trained_model_state_dict,
                'lottery_ticket': model.state_dict(),
                'original_initialization': original_initialization,
                'pruning_mask': pruning_mask
            }, output_file_path)

            # The iteration has finished
            self.logger.info('Finished iteration %d.', iteration)
