import torch
import torch.nn as nn

from transforms.transformation import Transformation
from transforms.affine import Affine

class PiecewiseAffineAffine(Transformation):
    def __init__(self):
        super().__init__(4)
        self.affine = Affine()

    def forward_trans(self, z, param):
        a, b, c, d = param[0], param[1], param[2], param[3]
        x = z - b

        x = torch.exp(a) * x * (x > 0) + x * (x <= 0)
        log_det_1 = torch.sum(a * (x > 0), dim=1)

        x += b
        
        x, log_det_2 = self.affine.forward_trans(x, [c,d])

        return x, (log_det_1 + log_det_2)

    def backward_trans(self, x, param):
        a, b, c, d = param[0], param[1], param[2], param[3]

        z, log_det_1 = self.affine.backward_trans(x, [c,d])
        z = z - b

        z = torch.exp(-a) * z * (z > 0) + z * (z <= 0)
        log_det_2 = torch.sum(-a * (z > 0), dim=1)

        z += b

        return z, (log_det_1 + log_det_2)
