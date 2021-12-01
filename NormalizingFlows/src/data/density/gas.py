from .processed_data import ProcessedData

import pandas as pd
import numpy as np
import torch

from os import path
from pathlib import Path
from ...utils import write_to_file

dirname = Path(__file__).parent.absolute()
DATAPATH_UNPROCESSED = str(dirname) + '/../../../data/unprocessed/gas/ethylene_CO.pickle'
DATAPATH_PREPROCESSED = str(dirname) + '/../../../data/preprocessed/gas/gas.hdf5'

class Gas(ProcessedData):
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
        data = pd.read_pickle(datapath)

        data.drop('Meth', axis=1, inplace=True)
        data.drop('Eth', axis=1, inplace=True)
        data.drop('Time', axis=1, inplace=True)
        
        correlation = data.corr()
        high_corr = (correlation > 0.98).to_numpy().sum(axis=1)

        while np.any(high_corr > 1):
            column_to_remove = np.where(high_corr > 1)[0][0]
            column_name = data.columns[column_to_remove]
            data.drop(column_name, axis=1, inplace=True)

            correlation = data.corr()
            high_corr = (correlation > 0.98).to_numpy().sum(axis=1)

        data = (data - data.mean())/data.std()
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)

        self.n, self.dim_input = data.shape
        self.valid_n = int(self.n * validation_perc)
        self.test_n = int(self.n * test_perc)
        self.train_n = self.n - self.valid_n - self.test_n 

        train_data = data[0:self.train_n].to_numpy()
        validation_data = data[self.train_n:(self.train_n+self.valid_n)].to_numpy()
        test_data = data[(self.train_n+self.valid_n):].to_numpy()

        self.train_data = torch.tensor(train_data)
        self.validation_data = torch.tensor(validation_data)
        self.test_data = torch.tensor(test_data)
