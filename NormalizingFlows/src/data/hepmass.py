from .data import Dataset

import torch
import pandas as pd
import numpy as np

from os import path
from pathlib import Path
from ..utils import write_to_file

dirname = Path(__file__).parent.absolute()
DATAPATH_UNPROCESSED = str(dirname) + '/../../data/unprocessed/hepmass/1000_'
DATAPATH_PREPROCESSED = str(dirname) + '/../../data/preprocessed/hepmass/hepmass.hdf5'

class Hepmass(Dataset):
    def __init__(self, validation_perc=0.1, preprocessed=True):
        super().__init__()
        
        if validation_perc != 0.1:
            preprocessed = False

        if preprocessed:
            if not path.isfile(DATAPATH_PREPROCESSED):
                self._get_unprocessed(DATAPATH_UNPROCESSED, validation_perc)
                write_to_file(DATAPATH_PREPROCESSED, self.train_data, self.validation_data, self.test_data)

            self._get_preprocessed(DATAPATH_PREPROCESSED)
        else:
            self._get_unprocessed(DATAPATH_UNPROCESSED, validation_perc)

    def _get_unprocessed(self, datapath, validation_perc):
        train_data = pd.read_csv(filepath_or_buffer=datapath+'train.csv', index_col=False)
        test_data = pd.read_csv(filepath_or_buffer=datapath+'test.csv', index_col=False)
        
        train_data = train_data[train_data[train_data.columns[0]]==1]
        train_data.drop(train_data.columns[0], axis=1, inplace=True)

        test_data = test_data[test_data[test_data.columns[0]]==1]
        test_data.drop(test_data.columns[0], axis=1, inplace=True)

        #test_data have one more attribute for some reason
        test_data.drop(test_data.columns[-1], axis=1, inplace=True)
        
        train_data = (train_data - train_data.mean())/(train_data.std())
        test_data = (test_data - test_data.mean())/(test_data.std())

        train_data, test_data = train_data.to_numpy(), test_data.to_numpy()

        self.dim_input = train_data.shape[1]

        features_to_remove = []
        for dim in range(self.dim_input):
            _, unique_count = np.unique(train_data[:, dim], return_counts=True, axis=0)
            min_nonunique_feat = np.min(unique_count)

            if min_nonunique_feat > 5:
                features_to_remove.append(dim)

        features_to_keep = [dim_index for dim_index in range(self.dim_input) if dim_index not in features_to_remove]
        train_data = train_data[:, features_to_keep]
        test_data = test_data[:, features_to_keep]

        self.dim_input = train_data.shape[1]
        self.valid_n = int(train_data.shape[0] * validation_perc)
        self.train_n = train_data.shape[0] - self.valid_n
        self.test_n = test_data.shape[0]

        self.train_data = torch.tensor(train_data[0:self.train_n,:])
        self.validation_data = torch.tensor(train_data[self.train_n:,:])
        self.test_data = torch.tensor(test_data)
