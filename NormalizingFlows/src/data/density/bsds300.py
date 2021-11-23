from .processed_data import ProcessedData

from os import path
from pathlib import Path
from ..utils import write_to_file

dirname = Path(__file__).parent.absolute()
DATAPATH_UNPROCESSED = str(dirname) + '/../../data/unprocessed/BSDS300/BSDS300.hdf5'
DATAPATH_PREPROCESSED = str(dirname) + '/../../data/preprocessed/BSDS300/BSDS300.hdf5'

class BSDS300(ProcessedData):
    def __init__(self, preprocessed=True):
        super().__init__()
        if preprocessed:
            if not path.isfile(DATAPATH_PREPROCESSED):
                self._get_preprocessed(DATAPATH_UNPROCESSED)
                write_to_file(DATAPATH_PREPROCESSED, self.train_data, self.validation_data, self.test_data)

            self._get_preprocessed(DATAPATH_PREPROCESSED)
        else:
            self._get_preprocessed(DATAPATH_UNPROCESSED)
