import numpy as np
import numpy.random as rnd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling.layerwise_sampling import LayerwiseSampling
from .samplingsets.samplingset_s import SamplingSetS

class MaskedLinear(nn.Linear):
    def __init__(self, layer_in, layer_out, bias):
        super().__init__(layer_in, layer_out, bias)
        '''
        Register buffer is saved as state_mask, which means it moves to 
        cuda/cpu with rest, but no gradient
        '''
        self.register_buffer('mask', torch.ones(layer_out, layer_in))
        self.layer_in = layer_in
        self.layer_out = layer_out

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class CONN(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_hidden, 
            dim_out, 
            c, 
            act_func=nn.ReLU(),
            plural=1, 
            bias=True,
            sample_set_generator=SamplingSetS(),
            mask_sampling=LayerwiseSampling(),
            input_resid=True,
            output_resid=True):

        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out 
        self.c = c
        self.act_func = act_func
        self.bias = bias
        self.plural = plural 
        self.sample_set_generator = sample_set_generator
        self.mask_sampling = mask_sampling
        self.input_resid = input_resid
        self.output_resid = output_resid

        if plural >= 2:
            self.c = self._muliply_input(plural)
        self.dim_net = [dim_in] + dim_hidden + [self.dim_out]

        self.sample_set = self.sample_set_generator.generate(self.c)
        self.m, self.masks = self.sample_new_masks()

        if output_resid:
            self._add_output_residuals()

        if input_resid:
            self._add_input_residuals()

        self.net = self._create_network()
        self.update_masks()

        self.net = nn.Sequential(*self.net)

        
    def _muliply_input(self, plural):
        new_c = []
        for i in self.c:
            for j in range(plural):
                new_c.append(i)
        self.dim_out = self.dim_out * plural
        return new_c

    def _create_network(self):
        net = []
        for l in range(len(self.masks)):
            layer_out, layer_in = self.masks[l].shape
            net.extend([
            MaskedLinear(layer_in, layer_out, self.bias),
            self.act_func,
            ])

        #Exclude activation function on output
        net.pop()

        return net

    def _add_input_residuals(self):
        d_0 = self.dim_net[0]
        for l in range(2, len(self.dim_net)-1):
            d_l = self.dim_net[l]
            zeros = np.zeros((d_l, d_0))
            self.masks[l-1] = np.concatenate((self.masks[l-1], zeros), axis=1)

        for l in range(1, len(self.masks)-1):
            mask = self.masks[l]
            zero_rows = np.where(np.sum(mask, axis=1) == 0)[0]
            for j in zero_rows:
                new_row = [{i}.issubset(self.m[l+1][j]) for i in range(self.dim_in)]
                mask[j,self.dim_net[l]:] = new_row

            self.masks[l] = mask

    def _add_output_residuals(self):
        connless_index = []
        resid_size = 0
        end_mask = self.masks[-1]

        for l in range(1, len(self.masks)-1):
            mask = self.masks[l]
            zero_cols = np.where(np.sum(mask, axis=0) == 0)[0]
            connless_index.append(zero_cols)
            for j in zero_cols:
                mask_connec = np.zeros((self.dim_out, 1))
                mask_connec[:,0] = [self.m[l][j].issubset(c_i) for c_i in self.c]
                end_mask = np.concatenate((end_mask, mask_connec), axis=1)
                resid_size += 1

        
        self.masks[-1] = end_mask
        
        self.connless_index = connless_index
        self.resid_size = resid_size

    def sample_new_masks(self):
        return self.mask_sampling.sample(self.c, self.sample_set, self.dim_net)

    def update_masks(self):
        masked_layers = [layer for layer in self.net if isinstance(layer, MaskedLinear)] 

        for layer, mask in zip(masked_layers, self.masks):
            layer.set_mask(mask)

    def forward(self, x):
        if self.output_resid and self.input_resid:
            return self._forward_with_resid(x)
        elif self.output_resid: 
            return self._forward_with_outresid(x)
        elif self.input_resid:
            return self._forward_with_inputresid(x)

        return self.net(x)

    def _forward_with_resid(self, x):
        outresid_ind = 0
        i = 0
        batch_size = x.size()[0]
        y = self.net[0](x)
        y = self.net[1](y)
        
        output_resids = torch.zeros_like(x.new(batch_size,self.resid_size))
        for l in range(2, len(self.net)-1, 2):
            y = torch.cat((y, x), dim=1)
            y = self.net[l](y)
            y = self.net[l+1](y)

            conn_ind = self.connless_index[i]
            output_resids[:,outresid_ind:outresid_ind+len(conn_ind)] = y[:,conn_ind]

            outresid_ind += len(conn_ind)
            i += 1

        y = torch.cat((y, output_resids), dim=1)
        return self.net[-1](y)

    def _forward_with_outresid(self, x):
        outresid_ind = 0
        i = 0
        batch_size = x.size()[0]
        y = self.net[0](x)
        y = self.net[1](y)
        
        output_resids = torch.zeros_like(x.new(batch_size,self.resid_size))
        for l in range(2, len(self.net)-1, 2):
            y = self.net[l](y)
            y = self.net[l+1](y)

            conn_ind = self.connless_index[i]
            output_resids[:,outresid_ind:outresid_ind+len(conn_ind)] = y[:,conn_ind]

            outresid_ind += len(conn_ind)
            i += 1

        y = torch.cat((y, output_resids), dim=1)
        return self.net[-1](y)

    def _forward_with_inputresid(self, x):
        y = self.net[0](x)
        y = self.net[1](y)

        for l in range(2, len(self.net)-1, 2):
            y = torch.cat((y, x), dim=1)
            y = self.net[l](y)
            y = self.net[l+1](y)

        return self.net[-1](y)
