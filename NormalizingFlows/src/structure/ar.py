import numpy as np

import torch
import torch.nn as nn

from ..conn.made import MADE
from ..nets.made import MADE as MADE2
from ..nets.const_neural_net import ConstNeuralNet
from ..utils import permute_data, inv_permute_data


class AR(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_hidden, 
            transform,
            permutation,
            flow_forward=True, 
            act_func=nn.ReLU(),
            *args,
            **kwargs):

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = transform.get_param_count() * dim_in
        self.dim_hidden = dim_hidden
        self.flow_forward = flow_forward

        plural = transform.get_param_count()
        #if self.dim_in == 1:
        #    self.made_net = ConstNeuralNet(dim_in, dim_hidden, dim_out, act_func, bias)
        #else:
        self.made_net = MADE2(dim_in, dim_hidden, self.dim_out, act_func, natural_ordering=True, **kwargs)

        self.transform = transform
        self.permutation = permutation

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x):
        return self.forward_flow(x) if self.flow_forward else self.backward_flow(x)

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
