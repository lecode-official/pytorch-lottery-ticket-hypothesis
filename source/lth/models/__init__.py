"""A sub-package that contains multiple neural network models."""

import os
import glob
import inspect
from enum import Enum
from typing import Callable, Union

import torch


def model_id(new_id: str) -> Callable[[type], type]:
    """A decorator, which adds a model ID to a model class.

    Args:
        new_id (str): The ID that is to be added to the model class.

    Returns:
        Callable[[type], type]: Returns the decorated model class.
    """

    def decorator(model_class: type) -> type:
        model_class.model_id = new_id
        return model_class
    return decorator


class LayerKind(Enum):
    """Represents an enumeration, which contains values for the different type of layers in a neural network."""

    FULLY_CONNECTED = 1
    CONVOLUTION_2D = 2


class Layer:
    """Represents a single prunable layer in the neural network."""

    def __init__(
            self,
            name: str,
            kind: LayerKind,
            weights: torch.nn.Parameter,
            biases: torch.nn.Parameter,
            initial_weights: torch.Tensor,
            initial_biases: torch.Tensor,
            pruning_mask: torch.Tensor) -> None:
        """Initializes a new Layer instance.

        Args:
            name (str): The name of the layer.
            kind (LayerKind): The kind of the layer.
            weights (torch.nn.Parameter): The weights of the layer.
            biases (torch.nn.Parameter): The biases of the layer.
            initial_weights (torch.Tensor): A copy of the initial weights of the layer.
            initial_biases (torch.Tensor): A copy of the initial biases of the layer.
            pruning_mask (torch.Tensor): The current pruning mask of the layer.
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

    def __init__(self) -> None:
        """Initializes a new BaseModel instance. Since this is a base class, it should never be called directly."""

        # Invokes the constructor of the base class
        super().__init__()

        # Initializes some class members
        self.layers = None

    def initialize(self) -> None:
        """Initializes the model. It initializes the weights of the model using Xavier Normal (equivalent to Gaussian Glorot used in the original
        Lottery Ticket Hypothesis paper). It also creates an initial pruning mask for the layers of the model. These are initialized with all ones. A
        pruning mask with all ones does nothing. This method must be called by all sub-classes at the end of their constructor.
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
                layer_kind = LayerKind.FULLY_CONNECTED
            elif isinstance(layer_module, torch.nn.Conv2d):
                layer_kind = LayerKind.CONVOLUTION_2D

            # If the layer is not one of the supported layers, then it is skipped
            if layer_kind is None:
                continue

            # Gets the weights and biases of the layer
            weights = None
            biases = None
            for parameter_name, parameter in self.named_parameters():
                if parameter_name == f'{layer_name}.weight':
                    weights = parameter
                if parameter_name == f'{layer_name}.bias':
                    biases = parameter

            # Initializes the weights of the layer using the Xavier Normal initialization method
            torch.nn.init.xavier_normal_(weights)

            # Stores a copy of the initial weights and biases, which are needed by the Lottery Ticket Hypothesis, because after each iteration, the
            # weights are reset to their respective initial values
            initial_weights = torch.empty_like(weights).copy_(weights)  # pylint: disable=no-member
            initial_biases = torch.empty_like(biases).copy_(biases)  # pylint: disable=no-member

            # Initializes the pruning masks of the layer, which are used for pruning as well as freezing the pruned weights during training
            pruning_mask = torch.ones_like(weights, dtype=torch.uint8)  # pylint: disable=no-member

            # Adds the layer to the internal list of layers
            self.layers.append(Layer(layer_name, layer_kind, weights, biases, initial_weights, initial_biases, pruning_mask))

    def get_layer_names(self) -> list[str]:
        """Retrieves the internal names of all the layers of the model.

        Returns:
            list[str]: Returns a list of all the names of the layers of the model.
        """

        layer_names = []
        for layer in self.layers:
            layer_names.append(layer.name)
        return layer_names

    def get_layer(self, layer_name: str) -> Layer:
        """Retrieves the layer of the model with the specified name.

        Args:
            layer_name (str): The name of the layer that is to be retrieved.

        Raises:
            LookupError: If the layer does not exist, an exception is raised.

        Returns:
            Layer: Returns the layer with the specified name.
        """

        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        raise LookupError(f'The specified layer "{layer_name}" does not exist.')

    def update_layer_weights(self, layer_name: str, new_weights: torch.Tensor) -> None:
        """Updates the weights of the specified layer.

        Args:
            layer_name (str): The name of the layer whose weights are to be updated.
            new_weights (torch.Tensor): The new weights of the layer.
        """

        self.state_dict()[f'{layer_name}.weight'].copy_(new_weights)

    def reset(self) -> None:
        """Resets the model back to its initial initialization."""

        for layer in self.layers:
            self.state_dict()[f'{layer.name}.weight'].copy_(layer.initial_weights)
            self.state_dict()[f'{layer.name}.bias'].copy_(layer.initial_biases)

    def move_to_device(self, device: Union[int, str, torch.device]) -> None:  # pylint: disable=no-member
        """Moves the model to the specified device.

        Args:
            device (Union[int, str, torch.device]): The device that the model is to be moved to.
        """

        # Moves the model itself to the device
        self.to(device)

        # Moves the initial weights, initial biases, and the pruning masks also to the device
        for layer in self.layers:
            layer.initial_weights = layer.initial_weights.to(device)
            layer.initial_biases = layer.initial_biases.to(device)
            layer.pruning_mask = layer.pruning_mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the neural network. Since this is the base model, the method is not implemented and must be implemented
        in all classes that derive from the base model.

        Args:
            x (torch.Tensor): The input to the neural network.

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: Returns the output of the neural network.
        """

        raise NotImplementedError()


def get_model_classes() -> list[type]:
    """Retrieves the classes of all the available models.

    Returns:
        list[type]: Returns a list containing the classes of all the models.
    """

    # Gets all the other Python modules that are in the models module
    model_modules = []
    for module_path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.py')):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        model_modules.append(__import__(f'lth.models.{module_name}', fromlist=['']))

    # Gets the model classes, which are all the classes in the models module and its sub-modules that inherit from BaseModel
    model_classes = []
    for module in model_modules:
        for _, module_class in inspect.getmembers(module, inspect.isclass):
            if BaseModel in module_class.__bases__ and module_class not in model_classes:
                model_classes.append(module_class)

    # Returns the list of model classes
    return model_classes


def get_model_ids() -> list[str]:
    """Retrieves the IDs of all available models.

    Returns:
        list[str]: Returns a list containing the IDs of all available models.
    """

    # Gets the IDs of all the models and returns them
    model_ids = []
    for model_class in get_model_classes():
        if hasattr(model_class, 'model_id'):
            model_ids.append(model_class.model_id)
    return model_ids


def create_model(id_of_model: str, input_size: tuple, number_of_input_channels: int, number_of_classes: int) -> BaseModel:
    """Creates the model with the specified name.

    Args:
        id_of_model (str): The ID of the model that is to be created.
        input_size (tuple): A tuple containing the edge lengths of the input tensors, which is the input size of the neural network.
        number_of_input_channels (int): The number of channels that the input tensor has.
        number_of_classes (int): The number of classes that the neural network should be able to differentiate. This corresponds to the output size of
            the neural network.

    Raises:
        ValueError: When the model with the specified name could not be found, an exception is raised.

    Returns:
        BaseModel: Returns the model with the specified name.
    """

    # Finds the class for the specified model, all models in this module must have a class-level variable containing a model identifier
    found_model_class = None
    for model_class in get_model_classes():
        if hasattr(model_class, 'model_id') and model_class.model_id == id_of_model:
            found_model_class = model_class
    if found_model_class is None:
        raise ValueError(f'The model with the name "{id_of_model}" could not be found.')

    # Creates the model and returns it
    return found_model_class(input_size, number_of_input_channels, number_of_classes)
