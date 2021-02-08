import numpy as np

import torch
import torch.nn as nn

from src.transforms.transformation import Transformation

class SigmoidPos(Transformation):
    def __init__(self):
        super().__init__(0)

    def forward_trans(self, z):
        x = 2 * (torch.sigmoid(z) - 0.5)
        log_det = z - 2*torch.log(1+torch.exp(z)) 
        log_det += torch.log(2*torch.ones_like(z))

        return x, log_det

    def backward_trans(self, x):
        temp = x/2 + 0.5
        z = torch.log(temp/(1-temp))
        log_det = -torch.log(1-x**2)
        log_det += torch.log(2*torch.ones_like(x))

        return z, log_det
