import numpy as np

import torch
import torch.nn as nn

from src.made import MADE


class MAF(nn.Module):
    def __init__(self, dim_in, dim_hidden, kl_forward=True, act_func=nn.ReLU()):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.kl_forward = kl_forward

        self.made_net = MADE(dim_in, dim_hidden, 2 * dim_in, act_func, True)

    def forward(self, x):
        return self.backward_flow(x) if self.kl_forward else self.forward_flow(x)

    def forward_flow(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        
        for d in range(self.dim_in):
            res = self.made_net(x)
            sigma, mu = res.split(self.dim_in, dim=1)

            x[:,d] = z[:,d] * torch.exp(sigma[:,d]) + mu[:,d]
            log_det += sigma[:,d]

        return x, log_det

    def backward_flow(self, x):
        res = self.made_net(x)
        sigma, mu = res.split(self.dim_in, dim=1)
        z = (x - mu) * torch.exp(-sigma)
        log_det = torch.sum(-sigma, dim=1)
        return z, log_det

