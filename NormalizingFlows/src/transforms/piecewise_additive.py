import torch
import torch.nn as nn

from .transformation import Transformation
from .affine import Affine

class PiecewiseAffineAdditive(Transformation):
    def __init__(self):
        super().__init__(3)

    def forward_trans(self, z, param):
        a, b, c = param[0], param[1], param[2] 
        x = z - c
        x = x - b

        x = torch.exp(a) * x * (x > 0) + x * (x <= 0)
        log_det = torch.sum(a * (x > 0), dim=1)

        x += b


        return x, log_det 

    def backward_trans(self, x, param):
        a, b, c = param[0], param[1], param[2] 

        #print(x)
        z = x - c 
        z = z - b
        #print(z)

        z = torch.exp(-a) * z * (z > 0) + z * (z <= 0)
        log_det = torch.sum(-a * (z > 0), dim=1)

        z += b

        #print(z)
        #print(torch.exp(-a),b,c)

        return z, log_det
