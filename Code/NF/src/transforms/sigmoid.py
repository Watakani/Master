import numpy as np

import torch
import torch.nn as nn

from src.transforms.transformation import Transformation

class Sigmoid(Transformation):
    def __init__(self):
        super().__init__(0)

    def forward_trans(self, z):
        x = torch.sigmoid(z)
        log_det = z - 2*torch.log(1+torch.exp(z))

        return x, log_det

    def backward_trans(self, x):
        z = torch.log(x/(1-x))
        log_det = -(torch.log(1-x) + torch.log(x))

        return z, log_det
