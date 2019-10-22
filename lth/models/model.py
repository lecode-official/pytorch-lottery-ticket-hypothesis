"""Represents a module containing the base class for all the models."""

from enum import Enum

import torch

class LayerKind(Enum):
    """Represents an enumeration, which contains values for the different type of layers in a neural network."""

    fully_connected = 1
    convolution_2d = 2

class Layer:
    """Represents a single prunable layer in the neural network."""

    def __init__(self, name, kind, weights, biases, initial_weights, initial_biases, pruning_mask):
        """
        Initializes a new Layer instance.

        Parameters
        ----------
            name: str
                The name of the layer.
            kind: LayerKind
                The kind of the layer.
            weights: torch.nn.Parameter
                The weights of the layer.
            biases: torch.nn.Parameter
                The biases of the layer.
            initial_weights: torch.Tensor
                A copy of the initial weights of the layer.
            initial_biases: torch.Tensor
                A copy of the initial biases of the layer.
            pruning_mask: torch.Tensor
                The current pruning mask of the layer.
        """

        self.name = name
        self.kind = kind
        self.weights = weights
        self.biases = biases
        self.initial_weights = initial_weights
        self.initial_biases = initial_biases
        self.pruning_mask = pruning_mask

class BaseModel(torch.nn.Module):
    """Represents the base class for all models."""

    def __init__(self):
        """Initializes a new BaseModel instance. Since this is a base class, it should never be called directly."""

        # Invokes the constructor of the base class
        super(BaseModel, self).__init__()

        # Initializes some class members
        self.layers = None

    def initialize(self):
        """
        Initializes the model. It initializes the weights of the model using Xavier Normal (equivalent to Gaussian Glorot used in the original Lottery
        Ticket Hypothesis paper). It also creates an initial pruning mask for the layers of the model. These are initialized with all ones. A pruning
        mask with all ones does nothing. This method must be called by all sub-classes at the end of their constructor.
        """

        # Gets the all the fully-connected and convolutional layers of the model (these are the only ones that are being used right now, if new layer
        # types are introduced, then they have to be added here, but right now all models only consist of these two types)
        self.layers = []
        for layer_name in self._modules:

            # Gets the layer module from the model
            layer_module = self._modules[layer_name]

            # Determines the type of the layer
            layer_kind = None
            if isinstance(layer_module, torch.nn.Linear):
                layer_kind = LayerKind.fully_connected
            elif isinstance(layer_module, torch.nn.Conv2d):
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

            # Initializes the weights of the layer using the Xavier Normal initialization method
            torch.nn.init.xavier_normal_(weights)

            # Stores a copy of the initial weights and biases, which are needed by the Lottery Ticket Hypothesis, because after each iteration, the
            # weights are reset to their respective initial values
            initial_weights = torch.empty_like(weights).copy_(weights)
            initial_biases = torch.empty_like(biases).copy_(biases)

            # Initializes the pruning masks of the layer, which are used for pruning as well as freezing the pruned weights during training
            pruning_mask = torch.ones_like(weights, dtype=torch.uint8)

            # Adds the layer to the internal list of layers
            self.layers.append(Layer(layer_name, layer_kind, weights, biases, initial_weights, initial_biases, pruning_mask))

    def get_layer_names(self):
        """
        Retrieves the internal names of all the layers of the model.

        Returns
        -------
            list
                Returns a list of all the names of the layers of the model.
        """

        layer_names = []
        for layer in self.layers:
            layer_names.append(layer.name)
        return layer_names

    def get_layer(self, layer_name):
        """
        Retrieves the layer of the model with the specified name.

        Parameters
        ----------
            layer_name: str
                The name of the layer that is to be retrieved.

        Raises
        ------
            LookupError
                If the layer does not exist, then a LookupError is raised.

        Returns
        -------
            Layer
                Returns the layer with the specified name.
        """

        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        raise LookupError('The specified layer "{0}" does not exist.'.format(layer_name))

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

    def reset(self):
        """Resets the model back to its initial initialization."""

        for layer in self.layers:
            self.state_dict()['{0}.weight'.format(layer.name)].copy_(layer.initial_weights)
            self.state_dict()['{0}.bias'.format(layer.name)].copy_(layer.initial_biases)

    def move_to_device(self, device):
        """
        Moves the model to the specified device.

        Parameters
        ----------
            device: torch.Device
                The device that the model is to be moved to.
        """

        # Moves the model itself to the device
        self.to(device)

        # Moves the initial weights, initial biases, and the pruning masks also to the device
        for layer in self.layers:
            layer.initial_weights = layer.initial_weights.to(device)
            layer.initial_biases = layer.initial_biases.to(device)
            layer.pruning_mask = layer.pruning_mask.to(device)

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
