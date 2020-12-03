import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class NormalizingFlow(nn.Module):
    def __init__(self, flows, base_distr, kl_forward=True):
        super().__init__()

        self.flows = nn.ModuleList(flows)
        self.base_distr = base_distr
        self.kl_forward = kl_forward 
        self.kl_backward = not kl_forward 

    def forward(self, x):
        return self.backward_flow(x) if self.kl_forward else self.forward_flow(x)

    def forward_flow(self, x):
        dim_row, _ = x.shape
        log_det = torch.zeros(dim_row)
        z = [x]
        z_i = x

        for flow in self.flows:
            z_i, log_det_i = flow(z_i)
            log_det += log_det_i
            z.append(z_i)

        return z, log_det

    def backward_flow(self, x):
        dim_row, _ = x.shape
        log_det = torch.zeros(dim_row)
        z = [x] 
        z_i = x

        for flow in self.flows[::-1]:
            z_i, log_det_i = flow(z_i)
            log_det += log_det_i
            z.append(z_i)
        
        return z[::-1], log_det

    def sample(self, dim):
        z_0 = self.base_distr.sample(dim)
        log_det = torch.zeros(dim[0])
        log_det += self.base_distr.log_prob(z_0)

        z = [z_0]
        z_i = z_0

        for flow in self.flows:
            z_i, log_det_i = flow.forward(z_i)
            log_det += log_det_i
            z.append(z_i)

        return z, log_det
    
    def evaluate(self, x):
        dim_row, _ = x.shape
        log_det = torch.zeros(dim_row)
        z = [x]
        z_i = x

        for flow in self.flows:
            z_i, log_det_i = flow.backward(z_i)
            log_det += log_det_i
            z.append(z_i)

        log_det += self.base_distr.log_prob(z[-1])
        return z[::-1], log_det




            




