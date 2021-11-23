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

    def log_prior(self, param):
        raise NotImplementedError

    def log_lik(self, param, data):
        raise NotImplementedError

    def _get_data(self, mode):
        if mode == 'test':
            return self.test_data.to(self.device)

        if mode == 'valid':
            return self.validation_data.to(self.device)

        return self.train_data.to(self.device)

    def get_training_data(self):
        return self.train_data.to(self.device)

    def get_validation_data(self):
        return self.validation_data.to(self.device)

    def get_test_data(self):
        return self.test_data.to(self.device)

    def evaluate(self, param, mode='train'):
        data = self._get_data(mode)
        return self.log_lik(param, data) + self.log_prior(param)

    def update_device(self, device):
        self.device = device
