"""Represents a module that contains the multiple neural network models based on the LeNet family of architectures first introduced by Yann LeCun."""

import torch

from . import BaseModel

class LeNet_300_100(BaseModel): # pylint: disable=invalid-name
    """Represents a much simpler LeNet variant, which has no convolutional layers."""

    # Specifies the ID by which the model can be identified
    model_id = 'lenet-300-100'

    # Specifies some known-to-work hyperparameters for the model (these are the same hyperparameters used by Frankle et al. in their original paper)
    learning_rate = 1.2e-3
    batch_size = 60
    number_of_epochs = 50

    def __init__(self, input_size=(28, 28), number_of_input_channels=1, number_of_classes=10):
        """
        Initializes a new LeNet5 instance.

        Parameters
        ----------
            input_size: int or tuple
                The size of the input of the neural network. Defaults to the typical MNIST size of 28x28.
            number_of_input_channels: int
                The number of channels that the input image has. Defaults to the typical MNIST number of channels: 1.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in MNIST: 10.
        """

        # Invokes the constructor of the base class
        super(LeNet_300_100, self).__init__()

        # Exposes some information about the model architecture
        self.name = 'LeNet-300-100'
        self.pruning_rates = {
            'fully_connected_1': 0.2,
            'fully_connected_2': 0.2,
            'fully_connected_3': 0.1
        }

        # Stores the arguments for later use
        if isinstance(input_size, tuple):
            self.input_size = 1
            for dimension in input_size:
                self.input_size *= dimension
        else:
            self.input_size = input_size
        self.number_of_input_channels = number_of_input_channels
        self.number_of_classes = number_of_classes

        # Creates the layers of the architecture
        self.fully_connected_1 = torch.nn.Linear(self.input_size * self.number_of_input_channels, 300)
        self.fully_connected_2 = torch.nn.Linear(300, 100)
        self.fully_connected_3 = torch.nn.Linear(100, number_of_classes)

        # Initializes the model
        self.initialize()

    def forward(self, x):
        """
        Performs the forward pass through the neural network.

        Parameters
        ----------
            x: torch.Tensor
                The input to the neural network.

        Returns
        -------
            torch.Tensor
                Returns the output of the neural network.
        """

        # Brings the input to the correct size
        x = x.view(x.size(0), self.input_size * self.number_of_input_channels)

        # Performs the forward pass through the neural network
        x = self.fully_connected_1(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_2(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_3(x)

        # Returns the result
        return x

class LeNet5(BaseModel):
    """
    Represents the classical convolutional neural network architecture LeNet5 introduced by Yann LeCun et al. in their paper "Gradient-Based Learning
    Applied to Document Recognition.", where it was used for character recognition.
    """

    # Specifies the ID by which the model can be identified
    model_id = 'lenet5'

    # Specifies some known-to-work hyperparameters for the model (these are the same hyperparameters used by Frankle et al. in their original paper)
    learning_rate = 1.2e-3
    batch_size = 60
    number_of_epochs = 50

    def __init__(self, input_size=(28, 28), number_of_input_channels=1, number_of_classes=10):
        """
        Initializes a new LeNet5 instance.

        Parameters
        ----------
            input_size: tuple
                A tuple containing the edge lengths of the input images, which is the input size of the first convolution of the neural network.
                Defaults to the typical MNIST size of 28x28. Be careful, this is not the input size described in the original paper. The original
                paper used an input size of 32x32, but since this model is mostly used to train on MNIST, the default should be be image size of the
                MNIST examples, which is 28x28.
            number_of_input_channels: int
                The number of channels that the input image has. Defaults to the typical MNIST number of channels: 1.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in MNIST: 10.
        """

        # Invokes the constructor of the base class
        super(LeNet5, self).__init__()

        # Exposes some information about the model architecture
        self.name = 'LeNet5'
        self.pruning_rates = {
            'convolution_1': 0.1,
            'convolution_2': 0.1,
            'fully_connected_1': 0.2,
            'fully_connected_2': 0.2,
            'fully_connected_3': 0.1
        }

        # Adds the first convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 5x5, the receptive field
        # shrinks by 4 on each side, after the convolution, a max pooling is applied with a filter size of 2x2, therefore, the receptive field shrinks
        # by a factor of 0.5, the edge length of the output after the first convolution and the max pooling is calculated by (x - 4) / 2, e.g.
        # convolution: (32, 32, 1) -> (28, 28, 6), avermaxage pooling: (28, 28, 6) -> (14, 14, 6)
        self.convolution_1 = torch.nn.Conv2d(number_of_input_channels, 6, kernel_size=5)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=6)
        output_size = ((input_size[0] - 4) // 2, (input_size[1] - 4) // 2)

        # Adds the second convolution layer followed by a BatchNorm layer, after which a second max pooling will be performed, the output size is
        # calculated exactly as in the first convolution layer, e.g. convolution: (14, 14, 6) -> (10, 10, 16), max pooling: (10, 10, 16) -> (5, 5, 16)
        self.convolution_2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=16)
        output_size = ((output_size[0] - 4) // 2, (output_size[1] - 4) // 2)

        # Adds three fully-connected layers to the end, the input size of the first layer will be the product of the edge lengths of the receptive
        # field of the second convolution layer (e.g. 5 * 5 = 25) multiplied by the number of feature maps in the second convolution (in this case the
        # number of feature maps in the second convolution is 16, so the input size, could for example be 5 * 5 * 16 = 400)
        self.fully_connected_1 = torch.nn.Linear(output_size[0] * output_size[1] * 16, 120)
        self.fully_connected_2 = torch.nn.Linear(120, 84)
        self.fully_connected_3 = torch.nn.Linear(84, number_of_classes)

        # Initializes the model
        self.initialize()

    def forward(self, x):
        """
        Performs the forward pass through the neural network.

        Parameters
        ----------
            x: torch.Tensor
                The input to the neural network.

        Returns
        -------
            torch.Tensor
                Returns the output of the neural network.
        """

        # Performs forward pass for the first convolutional layer
        x = self.convolution_1(x)
        x = self.batch_norm_1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Performs forward pass for the second convolutional layer
        x = self.convolution_2(x)
        x = self.batch_norm_2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Flattens the output of the second convolution layer so that is can be used as input for the first fully-connected layer
        x = x.view(x.size(0), -1)

        # Performs the forward pass through all fully-connected layers
        x = self.fully_connected_1(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_2(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_3(x)

        # Returns the result
        return x
