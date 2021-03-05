import numpy as np

import torch
import torch.nn as nn

from .transformation import Transformation

class Logit(Transformation):
    def __init__(self):
        super().__init__(0)

    def forward(self, x, forward=True):
        return self.forward_trans(x) if forward else self.backward_trans(x)

    def forward_trans(self, z):
        x = torch.log(z/(1-z))
        log_det = torch.sum(torch.log(-1/((z-1)*z)), dim=1)

        return x, log_det

    def backward_trans(self, x):
        z = torch.sigmoid(x)
        log_det = torch.sum(x - 2*torch.log(1+torch.exp(x)), dim=1)

        return z, log_det
