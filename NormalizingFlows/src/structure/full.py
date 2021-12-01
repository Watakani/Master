import torch
import torch.nn as nn

from ..nets.noisy_neural_net import NoisyNeuralNet



class Full(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            transform,
            flow_forward=True,
            act_func=nn.ReLU(),
            **args):

        super().__init__() 

        self.dim_in = dim_in
        self.dim_out = transform.get_param_count() * dim_in
        self.dim_hidden = dim_hidden
        self.flow_forward = flow_forward

        self.network = NoisyNeuralNet(dim_in, dim_hidden, self.dim_out, act_func, **args)
        
        self.transform = transform
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x):
        return self.forward_flow(x) if self.flow_forward else self.backward_flow(x)

    def forward_flow(self, z):
        res = self.network(z)
        param = res.split(self.dim_in, dim=1)

        x, log_det = self.transform(z, param, forward=True)

        return x, log_det

    def backward_flow(self, x):
        res = self.network(x)
        param = res.split(self.dim_in, dim=1)

        z, log_det = self.transform(x, param, forward=False)

        return z, log_det

    def update_device(self, device):
        self.device = device
