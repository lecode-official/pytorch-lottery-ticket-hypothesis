"""
Represents a module that contains the multiple neural network models based on the VGG family of architectures first introduced by K. Simonyan and A.
Zisserman in their paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". VGG was named after Oxford's renowned Visual Geometry
Group (VGG).
"""

import torch

from . import model_id
from . import BaseModel

@model_id('vgg2')
class Vgg2(BaseModel):
    """
    Represents a very small VGG-variant with only two convolution layers. In the original paper by Frankle et al., this is referred to as Conv-2.
    """

    def __init__(self, input_size=(32, 32), number_of_input_channels=3, number_of_classes=10):
        """
        Initializes a new Vgg2 instance.

        Parameters
        ----------
            input_size: tuple
                A tuple containing the edge lengths of the input images, which is the input size of the first convolution of the neural network.
                Defaults to the typical CIFAR-10 size of 32x32.
            number_of_input_channels: int
                The number of channels that the input image has. Defaults to the typical CIFAR-10 number of channels: 3.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in CIFAR-10: 10.
        """

        # Invokes the constructor of the base class
        super(Vgg2, self).__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG2'
        self.pruning_rates = {
            'convolution_1': 0.1,
            'convolution_2': 0.1,
            'fully_connected_1': 0.2,
            'fully_connected_2': 0.2,
            'fully_connected_3': 0.1
        }

        # Adds the first convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the first convolution remains the same, e.g. (32, 32, 3) ->
        # (32, 32, 64)
        self.convolution_1 = torch.nn.Conv2d(number_of_input_channels, 64, kernel_size=3, padding=1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=64)

        # Adds the second convolution layer followed by a BatchNorm layer, after the second convolution, a max pooling is applied with a filter size
        # of 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the second convolution is halved, e.g. (32, 32, 64) -> (16, 16, 64)
        self.convolution_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=64)
        output_size = (input_size[0] // 2, input_size[1] // 2)

        # Adds three fully-connected layers to the end, the input size of the first layer will be the product of the edge lengths of the receptive
        # field of the second convolution layer (e.g. 16 * 16 = 256) multiplied by the number of feature maps in the second convolution (in this case
        # the number of feature maps in the second convolution is 64, so the input size, could for example be 16 * 16 * 64 = 16.384)
        self.fully_connected_1 = torch.nn.Linear(output_size[0] * output_size[1] * 64, 256)
        self.fully_connected_2 = torch.nn.Linear(256, 256)
        self.fully_connected_3 = torch.nn.Linear(256, number_of_classes)

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

        # Performs forward pass for the second convolutional layer (the second convolutional layer is followed by a max pool)
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

@model_id('vgg4')
class Vgg4(BaseModel):
    """
    Represents a small VGG-variant with only four convolution layers. In the original paper by Frankle et al., this is referred to as Conv-4.
    """

    def __init__(self, input_size=(32, 32), number_of_input_channels=3, number_of_classes=10):
        """
        Initializes a new Vgg4 instance.

        Parameters
        ----------
            input_size: tuple
                A tuple containing the edge lengths of the input images, which is the input size of the first convolution of the neural network.
                Defaults to the typical CIFAR-10 size of 32x32.
            number_of_input_channels: int
                The number of channels that the input image has. Defaults to the typical CIFAR-10 number of channels: 3.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in CIFAR-10: 10.
        """

        # Invokes the constructor of the base class
        super(Vgg4, self).__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG4'
        self.pruning_rates = {
            'convolution_1': 0.1,
            'convolution_2': 0.1,
            'convolution_3': 0.1,
            'convolution_4': 0.1,
            'fully_connected_1': 0.2,
            'fully_connected_2': 0.2,
            'fully_connected_3': 0.1
        }

        # Adds the first convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the first convolution remains the same, e.g. (32, 32, 3) ->
        # (32, 32, 64)
        self.convolution_1 = torch.nn.Conv2d(number_of_input_channels, 64, kernel_size=3, padding=1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=64)

        # Adds the second convolution layer followed by a BatchNorm layer, after the second convolution, max pooling is applied with a filter size of
        # 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the second convolution is halved, e.g. (32, 32, 64) -> (16, 16, 64)
        self.convolution_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=64)
        output_size = (input_size[0] // 2, input_size[1] // 2)

        # Adds the third convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the third convolution remains the same, e.g. (16, 16, 64) ->
        # (16, 16, 128)
        self.convolution_3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=128)

        # Adds the fourth convolution layer followed by a BatchNorm layer, after the fourth convolution, max pooling is applied with a filter size of
        # 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the fourth convolution is halved, e.g. (16, 16, 128) -> (8, 8, 128)
        self.convolution_4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=128)
        output_size = (output_size[0] // 2, output_size[1] // 2)

        # Adds three fully-connected layers to the end, the input size of the first layer will be the product of the edge lengths of the receptive
        # field of the fourth convolution layer (e.g. 8 * 8 = 64) multiplied by the number of feature maps in the fourth convolution (in this case
        # the number of feature maps in the fourth convolution is 128, so the input size, could for example be 8 * 8 * 128 = 8.192)
        self.fully_connected_1 = torch.nn.Linear(output_size[0] * output_size[1] * 128, 256)
        self.fully_connected_2 = torch.nn.Linear(256, 256)
        self.fully_connected_3 = torch.nn.Linear(256, number_of_classes)

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

        # Performs forward pass for the second convolutional layer (the second convolutional layer is followed by a max pool)
        x = self.convolution_2(x)
        x = self.batch_norm_2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Performs forward pass for the third convolutional layer
        x = self.convolution_3(x)
        x = self.batch_norm_3(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the fourth convolutional layer (the fourth convolutional layer is followed by a max pool)
        x = self.convolution_4(x)
        x = self.batch_norm_4(x)
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

@model_id('vgg6')
class Vgg6(BaseModel):
    """
    Represents a small VGG-variant with only six convolution layers. In the original paper by Frankle et al., this is referred to as Conv-6.
    """

    def __init__(self, input_size=(32, 32), number_of_input_channels=3, number_of_classes=10):
        """
        Initializes a new Vgg6 instance.

        Parameters
        ----------
            input_size: tuple
                A tuple containing the edge lengths of the input images, which is the input size of the first convolution of the neural network.
                Defaults to the typical CIFAR-10 size of 32x32.
            number_of_input_channels: int
                The number of channels that the input image has. Defaults to the typical CIFAR-10 number of channels: 3.
            number_of_classes: int
                The number of classes that the neural network should be able to differentiate. This corresponds to the output size of the neural
                network, which defaults to the number of classes in CIFAR-10: 10.
        """

        # Invokes the constructor of the base class
        super(Vgg6, self).__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG6'
        self.pruning_rates = {
            'convolution_1': 0.1,
            'convolution_2': 0.1,
            'convolution_3': 0.1,
            'convolution_4': 0.1,
            'convolution_5': 0.1,
            'convolution_6': 0.1,
            'fully_connected_1': 0.2,
            'fully_connected_2': 0.2,
            'fully_connected_3': 0.1
        }

        # Adds the first convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the first convolution remains the same, e.g. (32, 32, 3) ->
        # (32, 32, 64)
        self.convolution_1 = torch.nn.Conv2d(number_of_input_channels, 64, kernel_size=3, padding=1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=64)

        # Adds the second convolution layer followed by a BatchNorm layer, after the second convolution, max pooling is applied with a filter size of
        # 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the second convolution is halved, e.g. (32, 32, 64) -> (16, 16, 64)
        self.convolution_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=64)
        output_size = (input_size[0] // 2, input_size[1] // 2)

        # Adds the third convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the third convolution remains the same, e.g. (16, 16, 64) ->
        # (16, 16, 128)
        self.convolution_3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=128)

        # Adds the fourth convolution layer followed by a BatchNorm layer, after the fourth convolution, max pooling is applied with a filter size of
        # 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the fourth convolution is halved, e.g. (16, 16, 128) -> (8, 8, 128)
        self.convolution_4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=128)
        output_size = (output_size[0] // 2, output_size[1] // 2)

        # Adds the fifth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the fifth convolution remains the same, e.g. (8, 8, 128) ->
        # (8, 8, 256)
        self.convolution_5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=256)

        # Adds the sixth convolution layer followed by a BatchNorm layer, after the sixth convolution, max pooling is applied with a filter size of
        # 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the sixth convolution is halved, e.g. (8, 8, 256) -> (4, 4, 256)
        self.convolution_6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=256)
        output_size = (output_size[0] // 2, output_size[1] // 2)

        # Adds three fully-connected layers to the end, the input size of the first layer will be the product of the edge lengths of the receptive
        # field of the sixth convolution layer (e.g. 4 * 4 = 16) multiplied by the number of feature maps in the sixth convolution (in this case
        # the number of feature maps in the sixth convolution is 256, so the input size, could for example be 4 * 4 * 256 = 4.096)
        self.fully_connected_1 = torch.nn.Linear(output_size[0] * output_size[1] * 256, 256)
        self.fully_connected_2 = torch.nn.Linear(256, 256)
        self.fully_connected_3 = torch.nn.Linear(256, number_of_classes)

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

        # Performs forward pass for the second convolutional layer (the second convolutional layer is followed by a max pool)
        x = self.convolution_2(x)
        x = self.batch_norm_2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Performs forward pass for the third convolutional layer
        x = self.convolution_3(x)
        x = self.batch_norm_3(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the fourth convolutional layer (the fourth convolutional layer is followed by a max pool)
        x = self.convolution_4(x)
        x = self.batch_norm_4(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Performs forward pass for the fifth convolutional layer
        x = self.convolution_5(x)
        x = self.batch_norm_5(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the sixth convolutional layer (the sixth convolutional layer is followed by a max pool)
        x = self.convolution_6(x)
        x = self.batch_norm_6(x)
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
