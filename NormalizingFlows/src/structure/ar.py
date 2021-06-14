import numpy as np

import torch
import torch.nn as nn

from ..conn.made import MADE
from ..nets.made import MADE as MADE2
from ..utils import permute_data, inv_permute_data


class AR(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_hidden, 
            transform,
            permutation,
            forward=True, 
            act_func=nn.ReLU(),
            **args):

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = transform.get_param_count() * dim_in
        self.dim_hidden = dim_hidden
        self.forward = forward

        plural = transform.get_param_count()
        self.made_net = MADE(dim_in, dim_hidden, dim_in, act_func, plural, **args)

        self.transform = transform
        self.permutation = permutation

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x):
        return self.foward_flow(x) if self.forward else self.backward_flow(x)

    def forward_flow(self, z):
        z = permute_data(z, self.permutation)

        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0], device=self.device)

        for d in range(self.dim_in):
            res = self.made_net(x.clone())
            res = res.split(self.dim_in, dim=1)
            param = [res_dim[:,d:(d+1)] for res_dim in res]

            x[:,d:(d+1)],log_det_dim = self.transform(z[:,d:(d+1)], param, forward=True)
            log_det = log_det + log_det_dim

        x = inv_permute_data(x, self.permutation)
        return x, log_det

    def backward_flow(self, x):
        x = permute_data(x, self.permutation) 

        res = self.made_net(x)
        param = res.split(self.dim_in, dim=1)

        z, log_det = self.transform(x, param, forward=False)

        z = inv_permute_data(z, self.permutation) 
        return z, log_det 

    def update_device(self, device):
        self.device = device
