"""Represents a module that contains the multiple neural network models based on the LeNet family of architectures."""

import torch
import torch.nn
import torch.nn.functional as functional

class LeNet_300_100(torch.nn.Module): # pylint: disable=invalid-name
    """Represents a much simpler LeNet variant, which has no convolutional layers."""

    def __init__(self, input_size=784, number_of_classes=10):
        """
        Initializes a new LeNet5 instance.

        Parameters
        ----------
            input_size: int
                The size of the input of the neural network. Defaults to the typical MNIST size of 28 * 28 = 784.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in MNIST: 10.
        """

        # Invokes the constructor of the base class
        super(LeNet_300_100, self).__init__()

        # Stores the arguments for later use
        self.input_size = input_size
        self.number_of_classes = number_of_classes

        # Creates the layers of the architecture
        self.fully_connected_1 = torch.nn.Linear(input_size, 300)
        self.fully_connected_2 = torch.nn.Linear(300, 100)
        self.fully_connected_3 = torch.nn.Linear(100, number_of_classes)

    def forward(self, x): # pylint: disable=arguments-differ
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
        x = x.view(x.size(0), self.input_size)

        # Performs the forward pass through the neural network
        x = self.fully_connected_1(x)
        x = functional.relu(x)
        x = self.fully_connected_2(x)
        x = functional.relu(x)
        x = self.fully_connected_3(x)

        # Returns the result
        return x

class LeNet5(torch.nn.Module):
    """
    Represents the classical convolutional neural network architecture LeNet5 introduced by Yann LeCun et al. in their paper "Gradient-Based Learning
    Applied to Document Recognition.", where it was used for character recognition.
    """

    def __init__(self, input_size=(32, 32), number_of_input_channels=1, number_of_classes=10):
        """
        Initializes a new LeNet5 instance.

        Parameters
        ----------
            input_size: tuple
                A tuple containing the edge lengths of the input images, which is the input size of the first convolution of the neural network.
                Defaults to the typical LeNet5 input size of 32x32. Be careful, this is the input size described in the original paper, but usually
                LeNet5 is used to train on MNIST, which has an image size of 28x28 and not 32x32.
            number_of_input_channels: int
                The number of channels that the input image has. Defaults to the typical MNIST number of channels: 1.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in MNIST: 10.
        """

        # Invokes the constructor of the base class
        super(LeNet5, self).__init__()

        # Adds the first convolution layer, since the convolution layer has a kernel size of 5x5, the receptive field shrinks by 4 on each side, after
        # the convolution, an average pooling is applied with a filter size of 2x2, therefore, the receptive field shrinks by a factor of 0.5, the
        # edge length of the output after the first convolution and the average pooling is calculated by (x - 4) / 2, e.g. convolution: (32, 32, 1) ->
        # (28, 28, 6), average pooling: (28, 28, 6) -> (14, 14, 6)
        self.convolution_1 = torch.nn.Conv2d(number_of_input_channels, 6, kernel_size=5)
        output_size = ((input_size[0] - 4) / 2, (input_size[1] - 4) / 2)

        # Adds the second convolution layer after which a second average pooling will be performed, the output size is calculated exactly as in the
        # first convolution layer, e.g. convolution: (14, 14, 6) -> (10, 10, 16), average pooling: (10, 10, 16) -> (5, 5, 16)
        self.convolution_2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        output_size = ((output_size[0] - 4) / 2, (output_size[1] - 4) / 2)

        # Adds three fully connected layers to the end, the input size of the first layer will be the product of the edge lengths of the receptive
        # field of the second convolution layer (e.g. 5 * 5 = 25) multiplied by the number of feature maps in the second convolution (in this case the
        # number of feature maps in the second convolution is 16, so the input size, could for example be 5 * 5 * 16 = 400)
        self.fully_connected_1 = torch.nn.Linear(output_size[0] * output_size[1] * 16, 120)
        self.fully_connected_2 = torch.nn.Linear(120, 84)
        self.fully_connected_3 = torch.nn.Linear(84, number_of_classes)

    def forward(self, x): # pylint: disable=arguments-differ
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
        x = functional.relu(x)
        x = functional.avg_pool2d(x, 2)

        # Performs forward pass for the second convolutional layer
        x = self.convolution_2(x)
        x = functional.relu(x)
        x = functional.avg_pool2d(x, 2)

        # Flattens the output of the second convolution layer so that is can be used as input for the first fully connected layer
        x = x.view(x.size(0), -1)

        # Performs the forward pass through all fully-connected layers
        x = self.fully_connected_1(x)
        x = functional.relu(x)
        x = self.fully_connected_2(x)
        x = functional.relu(x)
        x = self.fully_connected_3(x)

        # Returns the result
        return x
