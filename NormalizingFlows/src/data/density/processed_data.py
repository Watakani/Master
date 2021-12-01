from .data import Dataset
import h5py
import torch

class ProcessedData(Dataset):
    def __init__(self):
        super().__init__()

    def _get_preprocessed(self, datapath):
        datafile = h5py.File(datapath, 'r')

        self.train_data = torch.tensor(datafile['train'][:], dtype=torch.float)
        self.validation_data = torch.tensor(datafile['validation'][:], dtype=torch.float)
        self.test_data = torch.tensor(datafile['test'][:], dtype=torch.float)

        self.train_n, self.dim_input = self.train_data.size()
        self.valid_n = self.validation_data.size()[0]
        self.test_n = self.test_data.size()[0]

        datafile.close()
