from data.data import Dataset

from os import path
from utils import write_to_file

DATAPATH_UNPROCESSED = '../data/unprocessed/BSDS300/BSDS300.hdf5'
DATAPATH_PREPROCESSED = '../data/preprocessed/BSDS300/BSDS300.hdf5'

class BSDS300(Dataset):
    def __init__(self, preprocessed=True):
        super().__init__()
        if preprocessed:
            if not path.isfile(DATAPATH_PREPROCESSED):
                self._get_preprocessed(DATAPATH_UNPROCESSED)
                write_to_file(DATAPATH_PREPROCESSED, self.train_data, self.validation_data, self.test_data)

            self._get_preprocessed(DATAPATH_PREPROCESSED)
        else:
            self._get_preprocessed(DATAPATH_UNPROCESSED)
