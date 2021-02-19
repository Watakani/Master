import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class NormalizingFlow(nn.Module):
    def __init__(
            self, 
            flow,
            base_distr, 
            kl_forward=True):

        super().__init__()

        self.flow = nn.ModuleList(flow)
        self.base_distr = base_distr
        self.kl_forward = kl_forward 
        self.kl_backward = not kl_forward 

    def forward(self, x):
        return self.backward_flow(x) if self.kl_forward else self.forward_flow(x)

    def forward_flow(self, x):
        dim_row, _ = x.shape
        log_prob = torch.zeros(dim_row)
        log_prob += self.base_distr.log_prob(x)
        z = [x]
        z_i = x

        for f in self.flow:
            z_i, log_det_i = f(z_i)
            log_prob -= log_det_i
            z.append(z_i)

        return z, log_prob

    def backward_flow(self, x):
        dim_row, _ = x.shape
        log_prob = torch.zeros(dim_row)
        z = [x] 
        z_i = x

        for f in self.flow[::-1]:
            z_i, log_det_i = f(z_i)
            log_prob += log_det_i
            z.append(z_i)
        
        log_prob += self.base_distr.log_prob(z[-1])
        return z[::-1], log_prob

    def sample(self, n):
        z_0 = self.base_distr.sample((n,))
        log_prob = torch.zeros(n)
        log_prob += self.base_distr.log_prob(z_0)

        z = [z_0]
        z_i = z_0

        for f in self.flow:
            z_i, log_det_i = f.forward_flow(z_i)
            log_prob -= log_det_i
            z.append(z_i)

        return z, log_prob
    
    def evaluate(self, x):
        dim_row, _ = x.shape
        log_prob = torch.zeros(dim_row)
        z = [x]
        z_i = x

        for f in self.flow:
            z_i, log_det_i = f.backward_flow(z_i)
            log_prob += log_det_i
            z.append(z_i)

        log_prob += self.base_distr.log_prob(z[-1])
        return z[::-1], log_prob

    def get_base_distr(self):
        return self.base_distr
