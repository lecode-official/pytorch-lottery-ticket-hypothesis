"""Represents a module that contains a pruning methods that prune based on the magnitude of the weights."""

import logging

import tqdm
import torch


class LayerWiseMagnitudePruner:
    """
    Represents a pruning method that prunes away a certain layer-specific percentage of the weights of a neural network model that have the lowest
    magnitude. This is the pruning method used in the original paper by Frankle et al. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable
    Neural Networks".
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initializes a new LayerWiseMagnitudePruner instance.

        Parameters
        ----------
            model: torch.nn.Module
                The neural network model that is to be pruned.
        """

        self.model = model
        self.logger = logging.getLogger('lth.pruning.magnitude_pruning.LayerWiseMagnitudePruner')

    def create_pruning_masks(self) -> None:
        """
        Generates the pruning masks for all layers of the model. The pruning is not performed in-place on the layers of the model itself but a pruning
        mask is created for each layer (of the same shape as the layer), which has 0 values for all weights in the layer that were pruned and values
        of 1 for all weights that were not pruned.
        """

        # Creates the pruning masks for each layer of the model
        self.logger.info('Generating pruning mask for model %s...', self.model.name)
        for layer_name in tqdm.tqdm(self.model.get_layer_names(), unit='layer'):

            # Determines the pruning rate of the layer, if the pruning rate is 0.0, then no pruning is performed on the layer
            layer_pruning_rate = self.model.pruning_rates[layer_name]
            if layer_pruning_rate == 0.0:
                continue

            # Flattens the weights of the layer, because the sorting can only be performed for a single dimension
            layer = self.model.get_layer(layer_name)
            weights = layer.weights.reshape(-1)

            # Since the weights are pruned by their magnitude, the absolute values of the weights are retrieved
            weights = torch.abs(weights)  # pylint: disable=no-member

            # Sorts the weights ascending by their magnitude, this makes it easy to prune weights with the smallest magnitude, because the indices of
            # the weights with the smallest magnitude are at the beginning of the this array
            sorted_indices = torch.argsort(weights)  # pylint: disable=no-member

            # Determines the number of weights that should be pruned, since pruning is an iterative process of training, pruning, re-training, it
            # could be that the specified model was already sparsified previously, because the pruning strategy employed here is to sort the elements
            # by magnitude and then take the smallest n%, the same weights that are already zero would be pruned in any consecutive pruning,
            # therefore, the number of zeros in the layer are added to the number of pruned weights, otherwise no further pruning would occur
            number_of_zero_weights = weights.numel() - weights.nonzero().size(0)
            number_of_pruned_weights = int(layer_pruning_rate * (len(sorted_indices) - number_of_zero_weights)) + number_of_zero_weights

            # Creates the pruning mask which is 1 for all weights that are not pruned and
            pruning_mask = torch.zeros_like(weights, dtype=torch.uint8)  # pylint: disable=no-member
            pruning_mask[sorted_indices[:number_of_pruned_weights]] = 0
            pruning_mask[sorted_indices[number_of_pruned_weights:]] = 1

            # Reshapes the pruning mask to the original shape of the weights of the layer and stores it in the layer
            layer.pruning_mask = pruning_mask.reshape(layer.weights.shape)

        # Logs out a success message
        self.logger.info('Finished generating pruning mask for model %s.', self.model.name)

    def apply_pruning_masks(self) -> None:
        """Applies the pruning masks generated using create_pruning_masks. This is effectively the actual pruning."""

        # Applies the pruning masks for all layers
        total_number_of_weights = 0
        number_of_pruned_weights = 0
        number_of_zero_weights = 0
        self.logger.info('Applying pruning masks to the layers of the model...')
        for layer_name in tqdm.tqdm(self.model.get_layer_names(), unit='layer'):
            layer = self.model.get_layer(layer_name)
            total_number_of_weights += layer.weights.numel()
            pruned_weights = layer.weights * layer.pruning_mask
            number_of_pruned_weights += torch.sum(layer.pruning_mask == 0).item()  # pylint: disable=no-member
            number_of_zero_weights += pruned_weights.numel() - pruned_weights.nonzero().size(0)
            self.model.update_layer_weights(layer.name, pruned_weights)
        self.logger.info('Finished applying the pruning masks to the layers of the model.')
        self.logger.info(
            '%d of %d weights were pruned, sparsity of the model: %1.2f%%.',
            number_of_pruned_weights,
            total_number_of_weights,
            number_of_zero_weights / total_number_of_weights * 100
        )
