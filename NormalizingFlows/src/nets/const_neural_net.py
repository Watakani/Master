import torch
import torch.nn as nn


class ConstNeuralNet(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            dim_out,
            act_func=nn.ReLU(),
            bias=True):

        super().__init__() 

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.act_func = act_func

        assert self.dim_out % self.dim_in == 0

        self.net = []
        dim_net = [self.dim_in] + self.dim_hidden + [self.dim_out]

        for layer_in, layer_out in zip(dim_net[:-1], dim_net[1:]):
            self.net.extend([
                nn.Linear(layer_in, layer_out, bias),
                act_func,
                ])

        #Exclude actiavation function act_func on output
        self.net.pop()

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        temp = torch.ones_like(x)
        return self.net(temp)
