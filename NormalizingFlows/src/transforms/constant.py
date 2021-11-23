import torch
import torch.nn as nn

from .transformation import Transformation

class Constant(Transformation):

    def __init__(self, dim_in, forward_flow=True, a_param=torch.exp):
        super().__init__(2, forward_flow)
        self.a = nn.Parameter(torch.rand(1,dim_in))
        self.b = nn.Parameter(torch.rand(1, dim_in))
        self.a_param = a_param

    def training_direction(self, z):
        batch_size,_ = z.shape
        a, b = self.a_param(self.a), self.b

        x = z * a + b
        log_det = torch.ones(batch_size) * torch.sum(torch.log(a), dim=1)
        return x, log_det

    def inverse_direction(self, x):
        batch_size,_ = x.shape
        a, b = self.a_param(self.a), self.b
        
        z = (x - b)/a
        log_det = torch.ones(batch_size) * torch.sum(-torch.log(a), dim=1)
        
        return z, log_det
