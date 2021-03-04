from data.data import Dataset


import torch
import torch.nn
import h5py

DATAPATH = '../data/BSDS300/BSDS300.hdf5'

class BSDS300(Dataset):
    def __init__(self):
        super().__init__()

        datafile = h5py.File(DATAPATH, 'r')

        self.train_data = torch.from_numpy(datafile['train'][:])
        self.validation_data = torch.from_numpy(datafile['validation'][:])
        self.test_data = torch.from_numpy(datafile['test'][:])

        self.dim_input = self.train_data.size()[1]
        self.train_n = self.train_data.size()[0]
        self.valid_n = self.validation_data.size()[0]
        self.test_n = self.test_data.size()[0]

        datafile.close()

    def get_validation_data(self):
        return self.validation_data.to(self.device)
