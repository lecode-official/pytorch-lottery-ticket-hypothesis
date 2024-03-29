"""Represents a module, which contains hyperparameter defaults for several model and dataset combinations."""


def get_defaults(model_name: str, dataset_name: str, learning_rate: float, batch_size: int, number_of_epochs: int) -> tuple[float, int, int]:
    """Retrieves recommended hyperparameters for the specified model and dataset combination.

    Args:
        model_name (str): The name of the model for which the default hyperparameters are to be retrieved.
        dataset_name (str): The name of the dataset for which the default hyperparameters are to be retrieved.
        learning_rate (float): The learning rate specified by the user. When None, then a default value is returned, otherwise this is returned.
        batch_size (int): The batch size specified by the user. When None, then a default value is returned, otherwise this is returned.
        number_of_epochs (int): The number of epochs specified by the user. When None, then a default value is returned, otherwise this is returned.

    Raises:
        ValueError: If any of the hyperparameters were not specified and there is no a default value for the specified model and dataset combination,
            an exception is raised.

    Returns:
        tuple[float, int, int]: Returns a tuple containing a learning rate, a batch size, and the number of epochs to train for. The values are either
            the ones specified via the parameters or, if they are None, default values.
    """

    default_learning_rate, default_batch_size, default_number_of_epochs = None, None, None

    if model_name == 'lenet-300-100' and dataset_name == 'mnist':
        default_learning_rate, default_batch_size, default_number_of_epochs = 1.2e-3, 60, 50
    elif model_name == 'lenet-5' and dataset_name == 'mnist':
        default_learning_rate, default_batch_size, default_number_of_epochs = 1.2e-3, 60, 50
    elif model_name == 'conv-2' and dataset_name == 'cifar10':
        default_learning_rate, default_batch_size, default_number_of_epochs = 2e-4, 60, 20
    elif model_name == 'conv-4' and dataset_name == 'cifar10':
        default_learning_rate, default_batch_size, default_number_of_epochs = 3e-4, 60, 25
    elif model_name == 'conv-6' and dataset_name == 'cifar10':
        default_learning_rate, default_batch_size, default_number_of_epochs = 3e-4, 60, 30
    elif model_name == 'vgg19' and dataset_name == 'cifar10':
        default_learning_rate, default_batch_size, default_number_of_epochs = 3e-4, 64, 112

    learning_rate = learning_rate if learning_rate is not None else default_learning_rate
    batch_size = batch_size if batch_size is not None else default_batch_size
    number_of_epochs = number_of_epochs if number_of_epochs is not None else default_number_of_epochs

    if learning_rate is None:
        raise ValueError(f'No learning rate was specified and there are no defaults for training {model_name} on {dataset_name}.')
    if batch_size is None:
        raise ValueError(f'No batch size was specified and there are no defaults for training {model_name} on {dataset_name}.')
    if number_of_epochs is None:
        raise ValueError(f'No number of epochs was specified and there are no defaults for training {model_name} on {dataset_name}.')

    return learning_rate, batch_size, number_of_epochs
