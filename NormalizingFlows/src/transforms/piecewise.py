import torch
import torch.nn as nn

from .transformation import Transformation

class PiecewiseAffine(Transformation):
    def __init__(self):
        super().__init__(2)

    def forward_trans(self, z, param):
        a, b = param[0], param[1]
        x = z - b

        x = torch.exp(a) * x * (x > 0) + x * (x <= 0)
        log_det = torch.sum(a * (x > 0), dim=1)

        x += b

        return x, log_det

    def backward_trans(self, x, param):
        a, b = param[0], param[1]
        z = x - b

        z = torch.exp(-a) * z * (z > 0) + z * (z <= 0)
        log_det = torch.sum(-a * (z > 0), dim=1)

        z += b

        return z, log_det
