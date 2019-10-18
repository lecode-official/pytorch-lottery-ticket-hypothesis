"""Represents a module that contains helper classes to deal with layers of neural network models."""

from enum import Enum

import torch

class LayerKind(Enum):
    """Represents an enumeration, which contains values for the different type of layers in a neural network."""

    fully_connected = 1
    convolution_2d = 2

class Layer:
    """Represents a single prunable layer in the neural network."""

    def __init__(self, name, kind, weights):
        """
        Initializes a new Layer instance.

        Parameters
        ----------
            name: str
                The name of the layer.
            kind: LayerKind
                The kind of the layer.
            weights:
                The weights of the layer.
        """

        self.name = name
        self.kind = kind
        self.weights = weights

    @staticmethod
    def get_layers_from_model(model):
        """
        Retrieves the layers of the specified model.

        Parameters
        ----------
            model: torch.nn.Module
                The model whose layers are to be retrieved.

        Returns
        -------
            list
                Returns a list that contains the layers of the model.
        """

        # Gets the all the fully-connected and convolutional layers of the model (these are the only ones that are being pruned, if new layer types
        # are introduced, then they have to be added here, but right now all models only consist of these two types)
        layers = []
        for layer_name in model._modules: # pylint: disable=protected-access

            # Gets the layer object from the model
            layer = model._modules[layer_name] # pylint: disable=protected-access

            # Determines the type of the layer
            layer_kind = None
            if isinstance(layer, torch.nn.Linear):
                layer_kind = LayerKind.fully_connected
            elif isinstance(layer, torch.nn.Conv2d):
                layer_kind = LayerKind.convolution_2d

            # If the layer is not one of the supported layers, then it is skipped
            if layer_kind is None:
                continue

            # Gets the weights of the layer and
            for parameter_name, parameter in model.named_parameters():
                if parameter_name == '{0}.weight'.format(layer_name):
                    layers.append(Layer(layer_name, layer_kind, parameter))

        # Returns the layers that were retrieved
        return layers
