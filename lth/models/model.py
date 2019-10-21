"""Represents a module containing the base class for all the models."""

from enum import Enum

import torch

class LayerKind(Enum):
    """Represents an enumeration, which contains values for the different type of layers in a neural network."""

    fully_connected = 1
    convolution_2d = 2

class Layer:
    """Represents a single prunable layer in the neural network."""

    def __init__(self, name, kind, weights, biases):
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
            biases:
                The biases of the layer.
        """

        self.name = name
        self.kind = kind
        self.weights = weights
        self.biases = biases

class BaseModel(torch.nn.Module):
    """Represents the base class for all models."""

    def __init__(self):
        """Initializes a new BaseModel instance. Since this is a base class, it should never be called directly."""

        # Invokes the constructor of the base class
        super(BaseModel, self).__init__()

        # Initializes some class members
        self.layers = None
        self.initial_weights = None
        self.initial_biases = None
        self.pruning_masks = None

    def initialize(self):
        """
        Initializes the model. It initializes the weights of the model using Xavier Normal (equivalent to Gaussian Glorot used in the original Lottery
        Ticket Hypothesis paper). It also creates an initial pruning mask for the layers of the model. These are initialized with all ones. A pruning
        mask with all ones does nothing. This method must be called by all sub-classes at the end of their constructor.
        """

        # Gets the all the fully-connected and convolutional layers of the model (these are the only ones that are being pruned, if new layer types
        # are introduced, then they have to be added here, but right now all models only consist of these two types)
        self.layers = []
        for layer_name in self._modules:

            # Gets the layer object from the model
            layer = self._modules[layer_name]

            # Determines the type of the layer
            layer_kind = None
            if isinstance(layer, torch.nn.Linear):
                layer_kind = LayerKind.fully_connected
            elif isinstance(layer, torch.nn.Conv2d):
                layer_kind = LayerKind.convolution_2d

            # If the layer is not one of the supported layers, then it is skipped
            if layer_kind is None:
                continue

            # Gets the weights and biases of the layer
            weights = None
            biases = None
            for parameter_name, parameter in self.named_parameters():
                if parameter_name == '{0}.weight'.format(layer_name):
                    weights = parameter
                if parameter_name == '{0}.bias'.format(layer_name):
                    biases = parameter
            self.layers.append(Layer(layer_name, layer_kind, weights, biases))

        # Initializes the weights of the model using the Xavier Normal initialization method
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weights)

        # Stores a copy of the initial weights, which are needed by the Lottery Ticket Hypothesis, because after each iteration, the weights are reset
        # to their respective initial values
        self.initial_weights = {}
        self.initial_biases = {}
        for layer in self.layers:
            self.initial_weights[layer.name] = torch.empty_like(layer.weights).copy_(layer.weights)
            self.initial_biases[layer.name] = torch.empty_like(layer.biases).copy_(layer.biases)

        # Initializes the pruning masks of the layer, which are used for pruning as well as freezing the pruned weights during training
        self.pruning_masks = {}
        for layer in self.get_layers():
            self.pruning_masks[layer.name] = torch.ones_like(layer.weights, dtype=torch.uint8)

    def get_layers(self):
        """
        Retrieves the layers of the model.

        Returns
        -------
            list
                Returns a list that contains the layers of the model.
        """

        return self.layers

    def get_pruning_masks(self):
        """
        Retrieves the pruning masks of all layers of the model. The pruning masks are used for pruning and freezing weights of the model.

        Returns
        -------
            dict
                Returns a dictionary where the key is the name of the layer and the value is torch.Tensor representing the mask.
        """

        return self.pruning_masks

    def update_pruning_mask(self, layer_name, pruning_mask):
        """
        Updates the pruning mask of the specified layer.

        Parameters
        ----------
            layer_name: str
                The name of the layer for which the pruning mask is to be updated.
            pruning_mask: torch.Tensor
                The new pruning mask of the layer.

        Raises
        ------
            ValueError
                If the layer does not exist or the new pruning mask is in the wrong format, then a ValueError is raised.
        """

        # Validates the arguments
        if layer_name not in self.pruning_masks:
            raise ValueError('The specified layer "{0}" does not exist.'.format(layer_name))
        if pruning_mask.shape != self.pruning_masks[layer_name].shape:
            raise ValueError('The specified pruning mask has the wrong shape. Expected {0} but got {1}.'. format(
                str(pruning_mask.shape),
                str(self.pruning_masks[layer_name].shape)
            ))
        if pruning_mask.dtype != self.pruning_masks[layer_name].dtype:
            raise ValueError('The specified pruning mask has the wrong datatype. Expected {0} but got {1}.'. format(
                str(pruning_mask.dtype),
                str(self.pruning_masks[layer_name].dtype)
            ))

        # Stores the new pruning mask
        self.pruning_masks[layer_name] = pruning_mask

    def update_layer_weights(self, layer_name, new_weights):
        """
        Updates the weights of the specified layer.

        Parameters
        ----------
            layer_name: str
                The name of the layer whose weights are to be updated.
            new_weights: torch.Tensor
                The new weights of the layer.
        """

        self.state_dict()['{0}.weight'.format(layer_name)].copy_(new_weights)

    def reset_model(self):
        """Resets the model back to its initial initialization."""

        for layer in self.layers:
            self.state_dict()['{0}.weight'.format(layer.name)].copy_(self.initial_weights[layer.name])
            self.state_dict()['{0}.bias'.format(layer.name)].copy_(self.initial_biases[layer.name])

    def move_to_device(self, device):
        """
        Moves the model to the specified device.
        """

        # Moves the model itself to the device
        self.to(device)

        # Moves the initial weights, initial biases, and the pruning masks also to the device
        for layer in self.layers:
            self.initial_weights[layer.name] = self.initial_weights[layer.name].to(device)
            self.initial_biases[layer.name] = self.initial_biases[layer.name].to(device)
            self.pruning_masks[layer.name] = self.pruning_masks[layer.name].to(device)

    def forward(self, x):
        """
        Performs the forward pass through the neural network. Since this is the base model, the method is not implemented and must be implemented in
        all classes that derive from the base model.

        Parameters
        ----------
            x: torch.Tensor
                The input to the neural network.

        Returns
        -------
            torch.Tensor
                Returns the output of the neural network.
        """

        raise NotImplementedError()
