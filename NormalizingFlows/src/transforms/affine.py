import numpy as np

import torch
import torch.nn as nn

from .transformation import Transformation

class Affine(Transformation):
    def __init__(self, flow_forward=True, a_param=torch.exp):
        super().__init__(2, flow_forward)
        self.a_param = a_param

    def training_direction(self, z, param):
        a, b = param[0], param[1]
        a = self.a_param(a)

        x = z * a + b
        log_det = torch.sum(torch.log(a), dim=1)

        return x, log_det

    def inverse_direction(self, x, param):
        a, b = param[0], param[1]
        a = self.a_param(a)
         
        z = (x - b)/a
        log_det = torch.sum(-torch.log(a), dim=1)
        
        return z, log_det
