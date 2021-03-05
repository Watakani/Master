import torch
import h5py

from os import path
from ..utils import write_to_file

class Dataset:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def get_training_data(self):
        return self.train_data.to(self.device)

    def get_validation_data(self):
        return self.train_data.to(self.device)

    def get_test_data(self):
        return self.test_data.to(self.device)

    def update_device(self, device):
        self.device = device

    def _get_preprocessed(self, datapath):
        datafile = h5py.File(datapath, 'r')

        self.train_data = torch.tensor(datafile['train'][:])
        self.validation_data = torch.tensor(datafile['validation'][:])
        self.test_data = torch.tensor(datafile['test'][:])

        self.train_n, self.dim_input = self.train_data.size()
        self.valid_n = self.validation_data.size()[0]
        self.test_n = self.test_data.size()[0]

        datafile.close()
