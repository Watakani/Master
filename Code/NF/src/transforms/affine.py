import numpy as np

import torch
import torch.nn as nn

from src.transforms.transformation import Transformation

class Affine(Transformation):
    def __init__(self):
        super().__init__(2)

    def forward_trans(self, z, param):
        sigma, mu = param[0], param[1]
        x = z * torch.exp(sigma) + mu
        log_det = torch.sum(sigma, dim=1)

        return x, log_det

    def backward_trans(self, x, param):
        sigma, mu = param[0], param[1]
        z = (x - mu) * torch.exp(-sigma)
        log_det = torch.sum(-sigma, dim=1)

       return z, log_det
