import numpy as np
import numpy.random as rnd

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, layer_in, layer_out, bias):
        super().__init__(layer_in, layer_out, bias)
        '''
        Switched out and in dimension as nn.Linear uses x*A^T + b
        Register buffer is saved as state_mask, which means it moves to 
        cuda/cpu with rest, but no gradient
        '''
        self.register_buffer('mask', torch.ones(layer_out, layer_in))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class CONN(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_hidden, 
            dim_out, 
            c, 
            plural=1, 
            sample_set_generator,
            mask_sampling,
            bias=True,
            act_func==nn.ReLU()):

        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out 
        self.c = c
        self.plural = plural 
        self.sample_set_generator = sample_set_generator
        self.mask_sampling = mask_sampling
        self.bias = bias
        self.act_func = act_func

        if plural >= 2:
            self.c = _muliply_input(plural)

        self.net = _create_network()
        self.sample_set = self.sample_set_generator(c)
        
    def _muliply_input(self, plural):
        new_c = []
        for i in self.c:
            for j in range(plural):
                new_c.append(i)
        self.dim_out = self.dim_out * plural
        return new_c

    def _create_network(self):
        net = []
        self.dim_net = [self.dim_in] + self.dim_hidden + [self.dim_out]

        for layer_in, layer_out in zip(dim_net[:-1], dim_net[1:]):
            net.extend([
                MaskedLinear(layer_in, layer_out, bias),
                self.act_func,
                ])

        #Exclude activation function on output
        net.pop()

        return nn.Sequential(*net)

    def update_masks(self):
        masks = self.mask_sampling.sample(self.c, self.sample_set, self.dim_net)
        masked_layers = [layer for layer in self.net.modules() if isinstance(layer, MaskedLinear)] 

        for layer, mask in zip(masked_layers, masks):
            layer.set_mask(mask)

    def forward(self, x):
        return self.net(x)

