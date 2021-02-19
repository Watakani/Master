import torch
import torch.nn as nn

from transforms.transformation import Transformation

class PiecewiseAffine(Transformation):
    def __init__(self):
        super().__init__(2)

    def forward_trans(self, z, param):
        a, b = param[0], param[1]
        x = z - torch.exp(b)
        x = torch.exp(a) * torch.max(x, torch.zeros_like(x)) 
        x += torch.min(x, torch.zeros_like(x))
        log_det = torch.sum(a * (x > 0), dim=1)

        x += torch.exp(b)

        return x, log_det

    def backward_trans(self, x, param):
        a, b = param[0], param[1]
        z = x - torch.exp(b)

        z = torch.exp(-a) * torch.max(z, torch.zeros_like(z))
        z += torch.min(z, torch.zeros_like(z))
        log_det = torch.sum(-a * (z > 0), dim=1)

        z += torch.exp(b)

        return z, log_det
