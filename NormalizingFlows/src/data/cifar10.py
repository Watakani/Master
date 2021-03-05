from .data import Dataset

import torch
import torch.nn

from pathlib import Path

dirname = Path(__file__).parent.absolute()
DATAPATH_UNPROCESSED = str(dirname) + '../../data/unprocessed/cifar10/data_batch_'
DATAPATH_PROCESSED = str(dirname) + '../../data/preprocessed/cifar10/cifar10.hdf5'

class CIFAR10(Dataset):
    def __init__(self):
        pass

