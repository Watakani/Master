import numpy as np

import torch
import torch.nn as nn

from src.made import MADE


class AR(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_hidden, 
            dim_out, 
            permutation,
            transform,
            kl_forward=True, 
            act_func=nn.ReLU()):

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = transform.get_param_count() * dim_in
        self.dim_hidden = dim_hidden
        self.kl_forward = kl_forward
        self.made_net = MADE(dim_in, dim_hidden, dim_out, act_func)

        self.transform = transform


    def forward(self, x):
        x = x[:, permutation.permute()]
        x, log_det = self.backward_flow(x) if self.kl_forward else self.forward_flow(x)
        x = x[:, permutation.inv_permute()]
        return x, log_det

    def forward_flow(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.shape[0])

        for d in range(self.dim_in):
            res = self.made_net(z)
            res = res.split(self.dim_in, dim=1)
            param = [res_dim[:,d] for res_dim in res]

            z[:,d],log_det_dim = self.transform(x[:,d], param, forward=True)
            log_det += log_det_dim

        return z, log_det

    def backward_flow(self, z):
        res = self.made_net(z)
        param = res.split(self.dim_in, dim=1)

        x, log_det = self.transform(z, param, forward=False)

        return x, log_det
