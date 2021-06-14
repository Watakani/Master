import torch
import torch.nn as nn

from ..nets.neural_net import NeuralNet
from ..utils import permute_data, inv_permute_data


class TwoBlock(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            transform,
            permutation,
            flow_forward=True,
            act_func=nn.ReLU()):

        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.flow_forward = flow_forward
        self.param_count = transform.get_param_count()

        self.dim_1 = int(self.dim_in/2)
        self.dim_2 = self.dim_in - self.dim_1

        self.neural_net_1 = NeuralNet(self.dim_1, dim_hidden, 
                                        self.dim_2*self.param_count, act_func)
        self.neural_net_2 = NeuralNet(self.dim_2, dim_hidden, 
                                        self.dim_1*self.param_count, act_func)
        self.transform = transform
        self.permutation = permutation

    def forward(self, x):
        return self.forward_flow(x) if self.flow_forward else self.backward_flow(x)

    def forward_flow(self, z):
        z = permute_data(z, self.permutation)

        x = torch.zeros_like(z)

        res = self.neural_net_1(z[:,0:self.dim_1])
        param = res.split(self.dim_2, dim=1)
        x[:, self.dim_1:self.dim_in], log_det_1 = self.transform(
                z[:, self.dim_1:self.dim_in], param, forward=True)

        res = self.neural_net_2(x[:, self.dim_1:self.dim_in].clone())
        param = res.split(self.dim_1, dim=1)
        x[:, 0:self.dim_1], log_det_2 = self.transform(
                z[:, 0:self.dim_1], param, forward=True)

        x = inv_permute_data(x, self.permutation)
        return x, log_det_1 + log_det_2

    def backward_flow(self, x):
        x = permute_data(x, self.permutation)

        z = torch.zeros_like(x)

        res = self.neural_net_2(x[:, self.dim_1:self.dim_in])
        param = res.split(self.dim_1, dim=1)
        z[:, 0:self.dim_1], log_det_2 = self.transform(
                x[:, 0:self.dim_1], param, forward=False)

        res = self.neural_net_1(z[:, 0:self.dim_1].clone())
        param = res.split(self.dim_2, dim=1)
        z[:, self.dim_1:self.dim_in], log_det_1 = self.transform(
                x[:, self.dim_1:self.dim_in], param, forward=False)

        z = inv_permute_data(z, self.permutation)
        return z, log_det_1 + log_det_2

    def update_device(self, device):
        "Does not use any device"
        pass
