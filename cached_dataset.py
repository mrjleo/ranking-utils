import os
from os.path import split

import h5py
import numpy as np
from torch.utils.data import Dataset


class DatasetCache(Dataset):
    """
    Enables caching of data to a h5py file and can also act as torch dataset to read cached data.
    """

    def __init__(self, cache_path, input_specs, read_mode=False):
        """

        Args:
            cache_path {str} -- path to store the cached data to.
            input_specs {tuple} -- list of 3-tuples each specifying (name, shape, dtype) of an input. The last entry is
            expected to be the label specification.
            read_mode {bool} -- whether to use dataset for caching of for reading from cache.
        """
        self._read_mode = read_mode
        self.dataset_names = []

        if not read_mode:
            os.makedirs(split(cache_path)[0], exist_ok=True)
            self.cache_file = h5py.File(cache_path, 'w')
            self.cache_index = 0

            for spec in input_specs:
                self.cache_file.create_dataset(*spec)

        else:
            self.cache_file = h5py.File(cache_path, 'r')

        for spec in input_specs:
            self.dataset_names.append(spec[0])

    def add_to_cache(self, inputs):
        """Adds an entry to the dataset cache.

        Args:
            inputs {dict} -- a mapping from input names to their values
            label {float} -- an single float label for bce
        """
        assert len(inputs) == len(self.dataset_names)

        for name in self.dataset_names:
            self.cache_file[name][self.cache_index] = inputs[name]

        self.cache_index += 1

    def __getitem__(self, index):
        assert self._read_mode, 'can only read data from cache if read_mode=True'
        x = [self.cache_file[name][index] for name in self.dataset_names]
        inputs, label = x[:-1], x[-1]
        return inputs, label[np.newaxis]

    def __del__(self):
        self.cache_file.close()

    def __len__(self):
        k = list(self.cache_file.keys())[0]
        return len(self.cache_file[k])


class PairwiseDatasetCache(Dataset):
    """
    Enables caching of data to a h5py file and can also act as torch dataset to read cached data for pairwise input
    data.
    """

    def __init__(self, cache_path, pos_input_specs, neg_inputs_specs, read_mode=False):
        """

        Args:
            cache_path {str} -- path to store the cached data to.
            pos_input_specs {tuple} -- list of 3-tuples each specifying (name, shape, dtype) of an input.
            neg_inputs_specs {tuple} -- like pos_input_specs but for negative examples.
            read_mode {bool} -- whether to use dataset for caching of for reading from cache.
        """
        assert len(pos_input_specs) == len(neg_inputs_specs), 'positive and negative specs should have the same size'
        input_specs = pos_input_specs + neg_inputs_specs
        self.mid_idx = len(pos_input_specs)

        self._read_mode = read_mode
        self.dataset_names = []

        if not read_mode:
            os.makedirs(split(cache_path)[0], exist_ok=True)
            self.cache_file = h5py.File(cache_path, 'w')
            self.cache_index = 0

            for spec in input_specs:
                self.cache_file.create_dataset(*spec)

        else:
            self.cache_file = h5py.File(cache_path, 'r')

        for spec in input_specs:
            self.dataset_names.append(spec[0])

    def add_to_cache(self, inputs):
        """Adds an entry to the dataset cache.

        Args:
            inputs {dict} -- a mapping from input names to their values
            label {float} -- an single float label for bce
        """
        assert len(inputs) == len(self.dataset_names)

        for name in self.dataset_names:
            self.cache_file[name][self.cache_index] = inputs[name]

        self.cache_index += 1

    def __getitem__(self, index):
        assert self._read_mode, 'can only read data from cache if read_mode=True'
        x = [self.cache_file[name][index] for name in self.dataset_names]

        pos_inputs, neg_inputs = x[:self.mid_idx], x[self.mid_idx:]

        return pos_inputs, neg_inputs

    def __del__(self):
        self.cache_file.close()

    def __len__(self):
        k = list(self.cache_file.keys())[0]
        return len(self.cache_file[k])
