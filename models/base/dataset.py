import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from typing import Callable, Sequence, Any, Dict
import numpy as np
from typing import List, Tuple, Iterable
from sklearn.model_selection import KFold


class GenericDataLoader():
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.get_ids())

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_ids(self, i=None):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def split(self, train_ratio: float, seed=16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split dataset IDs into training and validation sets based on the given train ratio.

        Parameters:
            train_ratio (float): The proportion of the dataset to include in the training set.
            seed (int): Seed for the random number generator to ensure reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of training IDs and validation IDs.
        """
        if not seed or seed == 0:
            raise ValueError(
                "Seed must be a non-zero integer to ensure reproducibility.")

        ids = self.get_ids()  # Get all the ids
        train_size = int(train_ratio * len(ids))

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Randomly sample the train IDs without replacement
        train_ids = np.random.choice(ids, train_size, replace=False)

        # Get the remaining IDs for the validation set
        val_ids = np.setdiff1d(ids, train_ids)

        return self.post_split(train_ids, val_ids)

    def fold(self, k=5, shuffle=True):
        """
        Split dataset IDs into k folds for cross-validation.

        Parameters:
            k (int): Number of folds to create.

        Returns:
            A list of tuples containing the output of self.post_split for each fold.
            List[Tuple[GenericDataLoader, GenericDataLoader]]
        """
        ids = self.get_ids()
        # Set seed for reproducibility
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=16)
        return [self.post_split(train_ids, val_ids) for train_ids, val_ids in kf.split(ids)]

    def post_split(self, train_ids, val_ids):
        """
        Post-split method to be called after splitting the dataset.

        Parameters:
            train_ids (np.ndarray): Array of training IDs.
            val_ids (np.ndarray): Array of validation IDs.
        """
        return train_ids, val_ids
    
    def filter_ids(self, x: Callable[[Any], bool]):
        """
        Filter the ids in the dataset based on the given function.

        Parameters:
            x (Callable[[Any], bool]): A function that takes a file path and returns True if the file should be included.

        Returns:
            GenericDataLoader: A new DataLoader object containing only the files that satisfy the given condition.
        """
        raise NotImplementedError
        


class MockDataLoader(GenericDataLoader):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_ids(self):
        return np.arange(len(self.data))

    def post_split(self, train_ids, val_ids):
        if len(train_ids) == 0 or len(val_ids) == 0:
            return MockDataLoader([]), MockDataLoader([])
        return MockDataLoader(
            self.data[train_ids]), MockDataLoader(
            self.data[val_ids])


class Transformer:
    def __init__(self, train_transform, val_transform):
        self.train_transform = train_transform
        self.val_transform = val_transform

    def apply_train(self, **kwargs):
        raise NotImplementedError

    def apply_val(self, x):
        raise NotImplementedError


class GenericDataset(data.Dataset):
    '''
    A generic dataset class that can be used with any input and target data.
    Supports custom transformations for input and target data. Always returns 3 values: inputs, targets, *rest.
    *rest is a list of additional values that are not transformed or an empty Tuple if there are no additional values.
    The additional values are provided by the unpack method. To change the unpacking behavior, override the unpack method.
    Please note that the unpack method must always return 3 values: inputs, targets, *rest.
    '''

    def __init__(
            self, input_loader: GenericDataLoader,
            target_loader: GenericDataLoader,
            transform: Transformer):
        '''
        Create a new GenericDataset object.

        Parameters:
        ----------
        input_loader: A function that loads the input data. Must return a sequence of inputs.
        target_loader: A function that loads the target data. Must return a sequence of targets.
        transform: A dictionary containing transformations for input and target data.
        '''

        self.input_loader = input_loader
        self.target_loader = target_loader
        self.input_identifiers = input_loader.get_ids()
        self.target_identifiers = target_loader.get_ids()
        self.transform = transform

        if len(self.input_loader) != len(self.target_loader):
            raise ValueError(
                "Input and target datasets must have the same length.")

    def __len__(self) -> int:
        return len(self.input_loader)

    def unpack(self, inputs, targets):
        '''
        Unpack the input and target values if required

        Override this method if you need to unpack the inputs and targets in a specific way.
        The returned values should be in the order: inputs, targets, *rest
        The *rest values will be returned as additional values when the dataset is accessed.
        This is because we may need to return additional values along with the inputs and targets but we don't want to transform them.
        Ensuring that the *rest values are the last values returned will allow us to identify the transformed and non-transformed values.
        '''
        return inputs, targets, ()

    def __getitem__(self, index: int) -> Sequence[torch.Tensor]:
        inp = self.input_loader[index]
        target = self.target_loader[index]
        inp, target, *rest = self.unpack(inp, target)
        inp, target = self.transform(inp, target) if self.transform else (inp, target)

        return inp, target, rest
    
class DynamicDataset():
    def __init__(
            self, dataset: GenericDataset,
            generate:
            Callable
            [[np.ndarray, np.ndarray],
             Iterable[Tuple[np.ndarray, np.ndarray]]]):
        """
        Initialize the DynamicDataLoader.

        Parameters:
        dataset (GenericDataset): The dataset to generate data from.
        generate (Callable): Function that takes input and target arrays and returns an iterable of (input, target, *rest) tuples.
        """
        self.generate = generate
        self.dataset = dataset
        self.transform = dataset.transform
        dataset.transform = None # Take over the transformation responsibility

    def __iter__(self):
        """
        Iterate over the dataset and generate data using the provided generate function.
        """
        for input_data, target_data, *rest in self.dataset:
            for generated_input, generated_target, *rest in self.generate(
                    input_data, target_data, *rest):
                if self.transform:
                    generated_input, generated_target = self.transform(
                        generated_input, generated_target)
                yield generated_input, generated_target, rest


class GenericPredictionDataset(data.Dataset):
    '''
    A generic dataset class that can be used with any input data.
    Supports custom transformations for input data.
    '''

    def __init__(
            self, input_loader: GenericDataLoader,
            transform: Transformer):
        '''
        Create a new GenericPredictionDataset object.

        Parameters:
        ----------
        input_loader: A function that loads the input data. Must return a sequence of inputs.
        transform: A dictionary containing transformations for input data.
        '''

        self.input_loader = input_loader
        self.input_identifiers = input_loader.get_ids()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.input_loader)

    def __getitem__(self, index: int) -> Sequence[torch.Tensor]:
        inp = self.input_loader[index]
        inp, *rest = self.unpack(inp)
        inp = self.transform.apply_val(inp), rest

        return inp
