"""
Represents a module that contains the multiple neural network models based on the VGG family of architectures first introduced by K. Simonyan and A.
Zisserman in their paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". VGG was named after Oxford's renowned Visual Geometry
Group (VGG).
"""

import torch

from . import model_id
from . import BaseModel


@model_id('vgg5')
class Vgg5(BaseModel):
    """
    Represents a very small VGG-variant with only 5 weight layers. In the original paper by Frankle et al., this is referred to as Conv-2 as it has 2
    convolutional layers.
    """

    def __init__(self, input_size: tuple = (32, 32), number_of_input_channels: int = 3, number_of_classes: int = 10) -> None:
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
        super().__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG5'
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


@model_id('vgg7')
class Vgg7(BaseModel):
    """
    Represents a small VGG-variant with only 7 weight layers. In the original paper by Frankle et al., this is referred to as Conv-4, as it has 4
    convolutional layers.
    """

    def __init__(self, input_size: tuple = (32, 32), number_of_input_channels: int = 3, number_of_classes: int = 10) -> None:
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
        super().__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG7'
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Flattens the output of the fourth convolution layer so that is can be used as input for the first fully-connected layer
        x = x.view(x.size(0), -1)

        # Performs the forward pass through all fully-connected layers
        x = self.fully_connected_1(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_2(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_3(x)

        # Returns the result
        return x


@model_id('vgg9')
class Vgg9(BaseModel):
    """
    Represents a small VGG-variant with only 9 weight layers. In the original paper by Frankle et al., this is referred to as Conv-6, as it has 6
    convolutional layers.
    """

    def __init__(self, input_size: tuple = (32, 32), number_of_input_channels: int = 3, number_of_classes: int = 10) -> None:
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
        super().__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG9'
        self.pruning_rates = {
            'convolution_1': 0.15,
            'convolution_2': 0.15,
            'convolution_3': 0.15,
            'convolution_4': 0.15,
            'convolution_5': 0.15,
            'convolution_6': 0.15,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Flattens the output of the sixth convolution layer so that is can be used as input for the first fully-connected layer
        x = x.view(x.size(0), -1)

        # Performs the forward pass through all fully-connected layers
        x = self.fully_connected_1(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_2(x)
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_3(x)

        # Returns the result
        return x


@model_id('vgg17')
class Vgg17(BaseModel):
    """
    Represents a VGG-variant with 17 weight layers. In the original paper by Frankle et al. this is referred to as VGG19, because it is exactly as
    VGG19 with the difference, that this version was adapted to CIFAR-10 and is therefore missing 2 fully-connected layers at the end, but it has 16
    convolutional layers just as VGG19. Another difference to the original VGG19 is that after the last convolutional layer, an average pooling is
    performed instead of max pooling. This is the same as in the original paper by Frankle et al.
    """

    def __init__(self, input_size: tuple = (32, 32), number_of_input_channels: int = 3, number_of_classes: int = 10) -> None:
        """
        Initializes a new Vgg19 instance.

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
        super().__init__()

        # Exposes some information about the model architecture
        self.name = 'VGG17'
        self.pruning_rates = {
            'convolution_1': 0.2,
            'convolution_2': 0.2,
            'convolution_3': 0.2,
            'convolution_4': 0.2,
            'convolution_5': 0.2,
            'convolution_6': 0.2,
            'convolution_7': 0.2,
            'convolution_8': 0.2,
            'convolution_9': 0.2,
            'convolution_10': 0.2,
            'convolution_11': 0.2,
            'convolution_12': 0.2,
            'convolution_13': 0.2,
            'convolution_14': 0.2,
            'convolution_15': 0.2,
            'convolution_16': 0.2,
            'fully_connected_1': 0.0
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

        # Adds the sixth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1, the
        # receptive field does not shrink, i.e. the edge length of the output after the sixth convolution remains the same, e.g. (8, 8, 256) ->
        # (8, 8, 256)
        self.convolution_6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=256)

        # Adds the seventh convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1,
        # the receptive field does not shrink, i.e. the edge length of the output after the seventh convolution remains the same, e.g. (8, 8, 256) ->
        # (8, 8, 256)
        self.convolution_7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm_7 = torch.nn.BatchNorm2d(num_features=256)

        # Adds the eighth convolution layer followed by a BatchNorm layer, after the eighth convolution, max pooling is applied with a filter size of
        # 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the eighth convolution is halved, e.g. (8, 8, 256) -> (4, 4, 256)
        self.convolution_8 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm_8 = torch.nn.BatchNorm2d(num_features=256)
        output_size = (output_size[0] // 2, output_size[1] // 2)

        # Adds the ninth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1,
        # the receptive field does not shrink, i.e. the edge length of the output after the ninth convolution remains the same, e.g. (4, 4, 256) ->
        # (4, 4, 512)
        self.convolution_9 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch_norm_9 = torch.nn.BatchNorm2d(num_features=512)

        # Adds the tenth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1,
        # the receptive field does not shrink, i.e. the edge length of the output after the tenth convolution remains the same, e.g. (4, 4, 512) ->
        # (4, 4, 512)
        self.convolution_10 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_10 = torch.nn.BatchNorm2d(num_features=512)

        # Adds the eleventh convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1,
        # the receptive field does not shrink, i.e. the edge length of the output after the eleventh convolution remains the same, e.g. (4, 4, 512) ->
        # (4, 4, 512)
        self.convolution_11 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_11 = torch.nn.BatchNorm2d(num_features=512)

        # Adds the twelfth convolution layer followed by a BatchNorm layer, after the twelfth convolution, max pooling is applied with a filter size
        # of 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge length of the
        # output after the twelfth convolution is halved, e.g. (4, 4, 512) -> (2, 2, 512)
        self.convolution_12 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_12 = torch.nn.BatchNorm2d(num_features=512)
        output_size = (output_size[0] // 2, output_size[1] // 2)

        # Adds the thirteenth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of
        # 1, the receptive field does not shrink, i.e. the edge length of the output after the thirteenth convolution remains the same, e.g.
        # (2, 2, 512) -> (2, 2, 512)
        self.convolution_13 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_13 = torch.nn.BatchNorm2d(num_features=512)

        # Adds the fourteenth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of
        # 1, the receptive field does not shrink, i.e. the edge length of the output after the fourteenth convolution remains the same, e.g.
        # (2, 2, 512) -> (2, 2, 512)
        self.convolution_14 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_14 = torch.nn.BatchNorm2d(num_features=512)

        # Adds the fifteenth convolution layer followed by a BatchNorm layer, since the convolution layer has a kernel size of 3x3 and a padding of 1,
        # the receptive field does not shrink, i.e. the edge length of the output after the fifteenth convolution remains the same, e.g.
        # (2, 2, 512) -> (2, 2, 512)
        self.convolution_15 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_15 = torch.nn.BatchNorm2d(num_features=512)

        # Adds the sixteenth convolution layer followed by a BatchNorm layer, after the sixteenth convolution, average pooling is applied with a
        # filter size of 2x2, therefore, the receptive field shrinks by a factor of 0.5, since the kernel size is 3x3 and the padding is 1, the edge
        # length of the output after the sixteenth convolution is halved, e.g. (2, 2, 512) -> (1, 1, 512)
        self.convolution_16 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_16 = torch.nn.BatchNorm2d(num_features=512)
        output_size = (output_size[0] // 2, output_size[1] // 2)

        # Adds final fully-connected layer to the end, the input size of the layer will be the product of the edge lengths of the receptive field of
        # the sixteenth convolution layer (e.g. 1 * 1 = 1) multiplied by the number of feature maps in the sixteenth convolution (in this case the
        # number of feature maps in the sixteenth convolution is 512, so the input size, could for example be 1 * 1 * 512 = 512)
        self.fully_connected_1 = torch.nn.Linear(output_size[0] * output_size[1] * 512, number_of_classes)

        # Initializes the model
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Performs forward pass for the sixth convolutional layer
        x = self.convolution_6(x)
        x = self.batch_norm_6(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the seventh convolutional layer
        x = self.convolution_7(x)
        x = self.batch_norm_7(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the eighth convolutional layer (the eighth convolutional layer is followed by a max pool)
        x = self.convolution_8(x)
        x = self.batch_norm_8(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Performs forward pass for the ninth convolutional layer
        x = self.convolution_9(x)
        x = self.batch_norm_9(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the tenth convolutional layer
        x = self.convolution_10(x)
        x = self.batch_norm_10(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the eleventh convolutional layer
        x = self.convolution_11(x)
        x = self.batch_norm_11(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the twelfth convolutional layer (the twelfth convolutional layer is followed by a max pool)
        x = self.convolution_12(x)
        x = self.batch_norm_12(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Performs forward pass for the thirteenth convolutional layer
        x = self.convolution_13(x)
        x = self.batch_norm_13(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the fourteenth convolutional layer
        x = self.convolution_14(x)
        x = self.batch_norm_14(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the fifteenth convolutional layer
        x = self.convolution_15(x)
        x = self.batch_norm_15(x)
        x = torch.nn.functional.relu(x)

        # Performs forward pass for the sixteenth convolutional layer (the sixteenth convolutional layer is followed by an average pool)
        x = self.convolution_16(x)
        x = self.batch_norm_16(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.avg_pool2d(x, 2)

        # Flattens the output of the sixteenth convolution layer so that is can be used as input for the final fully-connected layer
        x = x.view(x.size(0), -1)

        # Performs the forward pass through the final fully-connected layer
        x = self.fully_connected_1(x)
        x = torch.nn.functional.relu(x)

        # Returns the result
        return x
