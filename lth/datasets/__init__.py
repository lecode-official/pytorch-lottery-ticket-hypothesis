"""Represents a module that contains multiple datasets for training neural networks."""

import os
import glob
import inspect

class BaseDataset:
    """Represents the base class for all datasets."""

def create_dataset(name, dataset_path, batch_size):
    """
    Creates the specified dataset.

    Parameters
    ----------
        name: str
            The name of the dataset that is to be created.
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

    # Gets all the other Python modules that are in the dataset module
    dataset_modules = []
    for module_path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.py')):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        dataset_modules.append(__import__('lth.datasets.{0}'.format(module_name), fromlist=['']))

    # Gets the dataset classes, which are all the classes in the dataset module and its sub-modules that inherit from BaseDataset
    dataset_classes = []
    for module in dataset_modules:
        for _, module_class in inspect.getmembers(module, inspect.isclass):
            if BaseDataset in module_class.__bases__ and module_class not in dataset_classes:
                dataset_classes.append(module_class)

    # Finds the class for the specified dataset, all datasets in this module must have a class-level variable containing a dataset identifier
    found_dataset_class = None
    for dataset_class in dataset_classes:
        if hasattr(dataset_class, 'dataset_id') and dataset_class.dataset_id == name:
            found_dataset_class = dataset_class
    if found_dataset_class is None:
        raise ValueError('The dataset with the name "{0}" could not be found.'.format(name))

    # Creates the dataset and returns it
    return found_dataset_class(dataset_path, batch_size)
