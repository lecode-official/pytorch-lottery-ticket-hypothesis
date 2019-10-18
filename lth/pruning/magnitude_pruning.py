"""Represents a module that contains a pruning methods that prune based on the magnitude of the weights."""

import torch

from .layers import Layer, LayerKind

class LayerWiseMagnitudePruner:
    """
    Represents a pruning method that prunes away a certain layer-specific percentage of the weights of a neural network model that have the lowest
    magnitude. This is the pruning method used in the original paper by Frankle et al. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable
    Neural Networks".
    """

    def __init__(self, model, fully_connected_pruning_rate, convolution_pruning_rate):
        """
        Initializes a new LayerWiseMagnitudePruner instance.

        Parameters
        ----------
            model: torch.nn.Module
                The neural network model that is to be pruned.
            fully_connected_pruning_rate: float
                The pruning rate that is to be used for fully-connected layers. The original paper uses 20% (i.e. a pruning rate of 0.2) for most
                models.
            convolution_pruning_rate: float
                The pruning rate that is to be used for convolutional layers. The original paper uses 10% (i.e. a pruning rate of 0.1) for most
                models.
        """

        self.model = model
        self.fully_connected_pruning_rate = fully_connected_pruning_rate
        self.convolution_pruning_rate = convolution_pruning_rate

    def prune(self):
        """
        Performs the layer-wise pruning on the model.

        Returns
        -------
            dict
                Returns a dictionary containing a pruning mask for each layer in the model. The pruning is not performed in-place on the layers of the
                model itself but a pruning mask is created for each layer (of the same shape as the layer), which is has 0 values for all weights in
                the layer that were pruned and values of 1 for all weights that were not pruned.
        """

        # Creates the pruning masks for each layer of the model
        masks = {}
        for layer in Layer.get_layers_from_model(self.model):

            # Flattens the weights of the layer, because the sorting can only be performed for a single dimension
            weights = layer.weights.reshape(-1)

            # Since the weights are pruned by their magnitude, the absolute values of the weights are retrieved
            weights = torch.abs(weights) # pylint: disable=no-member

            # Sorts the weights ascending by their magnitude, this makes it easy to prune weights with the smallest magnitude, because the indices of
            # the weights with the smallest magnitude are at the beginning of the this array
            sorted_indices = torch.argsort(weights) # pylint: disable=no-member

            # Determines the number of weights to prune based on the layer type
            if layer.kind == LayerKind.fully_connected:
                number_of_pruned_weights = int(self.fully_connected_pruning_rate * len(sorted_indices))
            else:
                number_of_pruned_weights = int(self.convolution_pruning_rate * len(sorted_indices))

            # Creates the pruning mask which is 1 for all weights that are not pruned and
            pruning_mask = torch.zeros_like(weights, dtype=torch.uint8) # pylint: disable=no-member
            pruning_mask[sorted_indices[:number_of_pruned_weights]] = 0
            pruning_mask[sorted_indices[number_of_pruned_weights:]] = 1

            # Reshapes the pruning mask to the original shape of the weights of the layer
            masks[layer.name] = pruning_mask.reshape(layer.weights.shape)

        # Returns the pruning masks for all layers
        return masks
