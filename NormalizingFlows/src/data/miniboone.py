from .data import Dataset

import torch
import numpy as np

from os import path
from pathlib import Path
from ..utils import write_to_file

dirname = Path(__file__).parent.absolute()
DATAPATH_UNPROCESSED = str(dirname) + '/../../data/unprocessed/miniboone/data.npy'
DATAPATH_PREPROCESSED = str(dirname) + '/../../data/preprocessed/miniboone/miniboone.hdf5'

class Miniboone(Dataset):
    def __init__(self, validation_perc=0.1, test_perc=0.1, preprocessed=True):
        super().__init__()

        if validation_perc != 0.1 or test_perc != 0.1:
            preprocessed = False

        if preprocessed:
            if not path.isfile(DATAPATH_PREPROCESSED):
                self._get_unprocessed(DATAPATH_UNPROCESSED, validation_perc, test_perc)
                write_to_file(DATAPATH_PREPROCESSED, self.train_data, self.validation_data, self.test_data)

            self._get_preprocessed(DATAPATH_PREPROCESSED)
        else:
            self._get_unprocessed(DATAPATH_UNPROCESSED, validation_perc, test_perc)

    def _get_unprocessed(self, datapath, validation_perc, test_perc):
        data = np.load(datapath)

        self.n, self.dim_input = data.shape
        self.valid_n = int(self.n * validation_perc)
        self.test_n = int(self.n * test_perc)
        self.train_n = self.n - self.valid_n - self.test_n

        train_data = data[0:self.train_n,:]
        validation_data = data[self.train_n:(self.train_n + self.valid_n),:]
        test_data = data[(self.train_n + self.valid_n):,:]

        train_and_validation = np.vstack((train_data, validation_data))

        mu = train_and_validation.mean(axis=0)
        std = train_and_validation.std(axis=0)

        train_data = (train_data - mu)/std
        validation_data = (validation_data - mu)/std
        test_data = (test_data - mu)/std
        
        self.train_data = torch.tensor(train_data)
        self.validation_data = torch.tensor(validation_data)
        self.test_data = torch.tensor(test_data)
