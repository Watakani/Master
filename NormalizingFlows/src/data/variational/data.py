import torch
import h5py

from os import path
from ...utils import write_to_file

class Dataset:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device_ = torch.device('cpu')
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def log_prior(self, param):
        raise NotImplementedError

    def log_lik(self, param, data):
        raise NotImplementedError

    def _get_data(self, mode):
        if mode == 'test':
            return [dat.to(self.device) for dat in self.test_data]

        if mode == 'valid':
            return [dat.to(self.device) for dat in self.validation_data]

        return [dat.to(self.device) for dat in self.train_data]

    def get_training_data(self):
        return [dat.to(self.device) for dat in self.train_data]

    def get_validation_data(self):
        return [dat.to(self.device) for dat in self.validation_data]

    def get_test_data(self):
        return [dat.to(self.device) for dat in self.test_data]
    
    def evaluate(self, param, mode='train'):
        data = self._get_data(mode)
        return self.log_lik(param.to(self.device_), [dat.to(self.device_) for dat in data]) + self.log_prior(param.to(self.device_))

    def update_device(self, device):
        self.device = device
