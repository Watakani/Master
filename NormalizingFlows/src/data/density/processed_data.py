from .data import Dataset

class ProcessedData(Dataset):
    def __init__(self):
        super().__init__()

    def _get_preprocessed(self, datapath):
        datafile = h5py.File(datapath, 'r')

        self.train_data = torch.tensor(datafile['train'][:])
        self.validation_data = torch.tensor(datafile['validation'][:])
        self.test_data = torch.tensor(datafile['test'][:])

        self.train_n, self.dim_input = self.train_data.size()
        self.valid_n = self.validation_data.size()[0]
        self.test_n = self.test_data.size()[0]

        datafile.close()
