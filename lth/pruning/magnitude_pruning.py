"""Represents a module that contains a pruning methods that prune based on the magnitude of the weights."""

import logging

import tqdm
import torch

from ..models.layers import Layer

class LayerWiseMagnitudePruner:
    """
    Represents a pruning method that prunes away a certain layer-specific percentage of the weights of a neural network model that have the lowest
    magnitude. This is the pruning method used in the original paper by Frankle et al. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable
    Neural Networks".
    """

    def __init__(self, model):
        """
        Initializes a new LayerWiseMagnitudePruner instance.

        Parameters
        ----------
            model: torch.nn.Module
                The neural network model that is to be pruned.
        """

        self.model = model
        self.logger = logging.getLogger('lth.pruning.magnitude_pruning.LayerWiseMagnitudePruner')

    def create_pruning_masks(self):
        """
        Generates the pruning masks for all layers of the model.

        Returns
        -------
            dict
                Returns a dictionary containing a pruning mask for each layer in the model. The pruning is not performed in-place on the layers of the
                model itself but a pruning mask is created for each layer (of the same shape as the layer), which is has 0 values for all weights in
                the layer that were pruned and values of 1 for all weights that were not pruned.
        """

        # Creates the pruning masks for each layer of the model
        masks = {}
        self.logger.info('Generating pruning mask for model %s...', self.model.name)
        for layer in tqdm.tqdm(Layer.get_layers_from_model(self.model), unit='layer'):

            # Determines the pruning rate of the layer, if the pruning rate is 0.0, then no pruning is performed on the layer
            layer_pruning_rate = self.get_layer_pruning_rate(layer)
            if layer_pruning_rate == 0.0:
                continue

            # Flattens the weights of the layer, because the sorting can only be performed for a single dimension
            weights = layer.weights.reshape(-1)

            # Since the weights are pruned by their magnitude, the absolute values of the weights are retrieved
            weights = torch.abs(weights) # pylint: disable=no-member

            # Sorts the weights ascending by their magnitude, this makes it easy to prune weights with the smallest magnitude, because the indices of
            # the weights with the smallest magnitude are at the beginning of the this array
            sorted_indices = torch.argsort(weights) # pylint: disable=no-member

            # Creates the pruning mask which is 1 for all weights that are not pruned and
            number_of_pruned_weights = int(layer_pruning_rate * len(sorted_indices))
            pruning_mask = torch.zeros_like(weights, dtype=torch.uint8) # pylint: disable=no-member
            pruning_mask[sorted_indices[:number_of_pruned_weights]] = 0
            pruning_mask[sorted_indices[number_of_pruned_weights:]] = 1

            # Reshapes the pruning mask to the original shape of the weights of the layer
            masks[layer.name] = pruning_mask.reshape(layer.weights.shape)

        # Returns the pruning masks for all layers
        self.logger.info('Finished generating pruning mask for model %s.', self.model.name)
        return masks

    def get_layer_pruning_rate(self, layer):
        """
        Determines the rate at which the specified layer should be pruned. For example a pruning rate of 0.2 means that 20% of the weights will be
        pruned.

        Parameters
        ----------
            layer: Layer
                The layer for which the pruning rate is to be determined.

        Returns
        -------
            float
                Returns the pruning rate of the specified layer. A pruning rate of 0.0 means that the layer should not be pruned at all.
        """

        # A model can specify model-wide pruning rate, which is to be applied to all layers, or it can specify pruning rates based on layer type, if
        # no pruning rate was specified for a layer kind, then it is assumed to be 0.0 (meaning that the layer should not be pruned at all)
        if isinstance(self.model.pruning_rates, float):
            return self.model.pruning_rates
        if isinstance(self.model.pruning_rates, dict) and layer.kind in self.model.pruning_rates:
            return self.model.pruning_rates[layer.king]
        return 0.0
