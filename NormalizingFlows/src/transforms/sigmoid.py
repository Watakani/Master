import numpy as np

import torch
import torch.nn as nn

from .transformation import Transformation

class Sigmoid(Transformation):
    def __init__(self):
        super().__init__(0)

    def forward_trans(self, z):
        x = torch.sigmoid(z)
        log_det = torch.sum(z - 2*torch.log(1+torch.exp(z)), dim=1)

        return x, log_det

    def backward_trans(self, ):
        z = torch.log(x/(1-x))
        log_det = torch.sum(torch.log(-1/((x-1)*x)))

        return z, log_det
