from data.data import Dataset

import torch
import torch.nn

DATAPATH = '../data/cifar10/data_batch_'

class CIFAR10(Dataset):
    def __init__(self):

