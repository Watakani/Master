import torch
import torch.nn as nn
import torch.nn.functional as F


from .transformation import Transformation

class PiecewiseAffine(Transformation):
    def __init__(self, forward_flow=True, a_param=torch.exp):
        super().__init__(2, forward_flow)
        self.a_param = a_param

    def training_direction(self, z, param):
        a, b = param[0], param[1]
        a = self.a_param(a)

        x = z - b

        x = a * x * (x > 0) + x * (x <= 0)
        log_det = torch.sum(torch.log(a) * (x > 0), dim=1)

        x += b

        return x, log_det

    def inverse_direction(self, x, param):
        a, b = param[0], param[1]
        a = self.a_param(a)

        z = x - b

        z = (z/a) * (z > 0) + z * (z <= 0)
        log_det = torch.sum(-torch.log(a) * (z > 0), dim=1)

        z += b

        return z, log_det
