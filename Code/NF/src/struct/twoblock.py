import torch
import torch.nn as nn

from src.neural_net import NeuralNet


class TwoBlock(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            transform,
            permutation,
            kl_forward=True,
            act_func=nn.ReLU()):

        super.__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.kl_forward = kl_forward
        self.param_count = transform.get_param_count()

        self.dim_in_1 = int(self.dim_in/2)
        self.dim_in_2 = self.dim_in - self.dim_in_1
        self.dim_out_1 = self.dim_in_2 
        self.dim_out_2 = self.dim_in_1 

        self.neural_net_1 = NeuralNet(self.dim_in_1, dim_hidden, 
                                        self.dim_out_1*self.param_count, act_func)
        self.neural_net_2 = NeuralNet(self.dim_in_2, dim_hidden, 
                                        self.dim_out_2*self.param_count, act_func)
        self.transform = transform

    def forward(self, x):
        x = x[:, permutation.permute()]
        x, log_det = self.backward_flow(x) if self.kl_forward else
                        self.forward_flow(x)
        x = x[:, permutation.inv_permute()]

        return x, log_det

    def forward_flow(self, z):
        res = self.neural_net_1(z[:,self.dim_in_1])
        param = res.split(self.dim_out_1, dim=1)
        x = torch.zeros_like(z)
        x[:, self.dim_out_1], log_det_1 = self.transform(
                                z[:, self.dim_out_1], param, forward=True)

        res = self.neural_net_2(x[:, self.dim_in_2])
        param = res.split(self.dim_out_2, dim=1)
        x[:, self.dim_out_2], log_det_2 = self.transform(
                                z[:, self.dim_out_2], param, forward=True)

        return x, (log_det_1 + log_det_2)

    def backward_flow(self, x):
        res = self.neural_net_2(x[:, self.dim_in_2])
        param = res.split(self.dim_out_2, dim=1)
        z = troch.zeros_like(x)
        z[:, self.dim_out_2], log_det_2 = self.transform(
                                x[:, self.dim_out_2], param, forward=False)

        res = self.neural_net_1(z[:, self.dim_in_1])
        param = res.split(self.dim_out_1, dim=1)
        z[:, self.dim_out_1], log_det_1 = self.transform(
                                x[:, self.dim_out_1], param, forward=False)

        return z, (log_det_1 + log_det_2)
