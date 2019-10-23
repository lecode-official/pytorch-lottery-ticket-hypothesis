"""Represents a module, which contains hyperparameter defaults for several model and dataset combinations."""

def get_defaults(model_name, dataset_name):
    """
    Retrieves recommended hyperparameters for the specified model and dataset combination.

    Parameters
    ----------
        model_name: str
            The name of the model for which the default hyperparameters are to be retrieved.
        dataset_name: str
            The name of the dataset for which the default hyperparameters are to be retrieved.

    Returns
    -------
        tuple
            Returns a tuple containing a learning rate, a batch size, and the number of epochs to train for.
    """

    if model_name == 'lenet-300-100' and dataset_name == 'mnist':
        return 1.2e-3, 60, 50
    if model_name == 'lenet5' and dataset_name == 'mnist':
        return 1.2e-3, 60, 50
    if model_name == 'vgg2' and dataset_name == 'cifar10':
        return 2e-4, 60, 20

    return None, None, None
