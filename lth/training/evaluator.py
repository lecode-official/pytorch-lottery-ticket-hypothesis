"""Evaluation of trained neural network models."""

import logging

import tqdm
import torch

class Evaluator:
    """Represents a standard evaluation procedure, which evaluates a model on the complete test split of the dataset and reports the accuracy."""

    def __init__(self, model, dataset):
        """
        Initializes a new Evaluator instance.

        Parameters
        ----------
            model: torch.nn.Module
                The neural network model that is to be evaluated.
            dataset
                The dataset on which the neural network model is to be evaluated.
        """

        self.model = model
        self.dataset = dataset
        self.logger = logging.getLogger('lth.training.evaluator.Evaluator')

    def evaluate(self):
        """Evaluates the model."""

        # Since we are only evaluating the model, the gradient does not have to be computed
        with torch.no_grad():

            # Checks if CUDA is available, in that case the evaluation is performed on the first GPU on the system, otherwise the CPU is used
            is_cuda_available = torch.cuda.is_available()
            device = torch.device('cuda:0' if is_cuda_available else 'cpu')

            # Transfers the model to the selected device
            self.model.to(device)

            # Puts the model into evaluation mode (this is important for some layers, like dropout and BatchNorm which have different bahavior during
            # training and evaluation)
            self.model.eval()

            # Cycles through the whole test split of the dataset and performs the evaluation
            self.logger.info('Evaluating the model...')
            correct_predictions = 0
            number_of_predictions = 0
            for batch in tqdm.tqdm(self.dataset.test_split, unit='batch'):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted_classes = torch.max(outputs, 1)
                correctness = (predicted_classes == labels).squeeze()
                for is_correct in correctness:
                    correct_predictions += is_correct.item()
                    number_of_predictions += 1

            # Computes the accuracy and reports it to the user
            accuracy = 100 * correct_predictions / number_of_predictions
            self.logger.info('Accuracy: %1.2f%%.', round(accuracy, 2))
            self.logger.info('Finished evaluating the model.')
