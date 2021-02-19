import numpy as np

import torch
import torch.nn as nn

from transforms.transformation import Transformation

class Logit(Transformation):
    def __init__(self):
        super().__init__(0)

    def forward_trans(self, z):
        x = torch.log(z/(1-z))
        log_det = -(torch.log(1-z) + torch.log(z))

        return x, log_det

    def backward_trans(self, x):
        z = torch.sigmoid(x)
        log_det = x - 2*torch.log(1+torch.exp(x))

        return z, log_det
