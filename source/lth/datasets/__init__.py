"""Represents a module that contains multiple datasets for training neural networks."""

import os
import glob
import inspect


def dataset_id(new_id):
    """A decorator, which adds a dataset ID to a dataset class."""

    def decorator(dataset_class):
        dataset_class.dataset_id = new_id
        return dataset_class
    return decorator


class BaseDataset:
    """Represents the base class for all datasets."""


def get_dataset_classes():
    """
    Retrieves the classes of all the available datasets.

    Returns
    -------
        list
            Returns a list containing the classes of all the datasets.
    """

    # Gets all the other Python modules that are in the dataset module
    dataset_modules = []
    for module_path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.py')):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        dataset_modules.append(__import__(f'lth.datasets.{module_name}', fromlist=['']))

    # Gets the dataset classes, which are all the classes in the dataset module and its sub-modules that inherit from BaseDataset
    dataset_classes = []
    for module in dataset_modules:
        for _, module_class in inspect.getmembers(module, inspect.isclass):
            if BaseDataset in module_class.__bases__ and module_class not in dataset_classes:
                dataset_classes.append(module_class)

    # Returns the list of dataset classes
    return dataset_classes


def get_dataset_ids():
    """
    Retrieves the IDs of all available datasets.

    Returns
    -------
        list
            Returns a list containing the IDs of all available datasets.
    """

    # Gets the IDs of all the datasets and returns them
    dataset_ids = []
    for dataset_class in get_dataset_classes():
        if hasattr(dataset_class, 'dataset_id'):
            dataset_ids.append(dataset_class.dataset_id)
    return dataset_ids


def create_dataset(id_of_dataset, dataset_path, batch_size):
    """
    Creates the specified dataset.

    Parameters
    ----------
        id_of_dataset: str
            The ID of the dataset that is to be created.
        path: str
            The path where the dataset is stored. If it does not exist, it is automatically downloaded to the specified location.
        batch_size: int
            The number of samples that are to be batched together.

    Raises
    ------
        ValueError
            When the dataset with the specified name could not be found, then a ValueError is raised.

    Returns
    -------
        BaseDataset
            Returns the dataset with the specified name.
    """

    # Finds the class for the specified dataset, all datasets in this module must have a class-level variable containing a dataset identifier
    found_dataset_class = None
    for dataset_class in get_dataset_classes():
        if hasattr(dataset_class, 'dataset_id') and dataset_class.dataset_id == id_of_dataset:
            found_dataset_class = dataset_class
    if found_dataset_class is None:
        raise ValueError(f'The dataset with the name "{id_of_dataset}" could not be found.')

    # Creates the dataset and returns it
    return found_dataset_class(dataset_path, batch_size)
