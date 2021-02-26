import numpy as np
import numpy.random as rnd
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, layer_in, layer_out, bias=True):
        super().__init__(layer_in, layer_out, bias)
        #Swithced out and in as nn.Linear uses x*A^T + b
        #Register buffer is saved as state_mask, moves to cuda/cpu with rest, but no gradient
        self.register_buffer('mask', torch.ones(layer_out, layer_in))
        #self.weight = nn.Parameter(self.weight * 0.01)

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_func=nn.ReLU(), natural_ordering=False):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.natural_ordering = natural_ordering

        assert self.dim_out % self.dim_in == 0

        self.net = []
        dim_net = [self.dim_in] + self.dim_hidden + [self.dim_out]

        for layer_in, layer_out in zip(dim_net[:-1], dim_net[1:]):
            self.net.extend([
                        MaskedLinear(layer_in, layer_out),
                        act_func,
                    ])

        #exclude activation function act_func on output
        self.net.pop()

        self.net = nn.Sequential(*self.net)

        self.m = {}
        self.update_masks()


    def update_masks(self):
        num_hidden = len(self.dim_hidden)

        self.m[-1] = np.arange(self.dim_in) if self.natural_ordering else rnd.permutation(self.dim_in)

        for layer in range(num_hidden):
            self.m[layer] = rnd.randint(self.m[layer-1].min(), self.dim_in-1, 
                                            self.dim_hidden[layer])

        #[:,None] makes array a row-vec, [None,:] a col-vec
        masks = [self.m[layer-1][:,None] <= self.m[layer][None,:] 
                            for layer in range(num_hidden)]

        masks.append(self.m[num_hidden-1][:,None] < self.m[-1][None,:])

        if self.dim_out > self.dim_in:
            k = self.dim_out // self.dim_in
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        layers = [layer for layer in self.net.modules() if isinstance(layer, MaskedLinear)] 
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    def forward(self, x):
        return self.net(x)
