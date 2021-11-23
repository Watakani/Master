import torch
import torch.nn as nn

class Transformation(nn.Module):
    def __init__(self, parameter_count, flow_forward=True):
        super().__init__()
        self.parameter_count = parameter_count
        self.flow_forward = flow_forward

    def forward(self, *x, forward=True):
        return self.forward_trans(*x) if forward else self.backward_trans(*x)

    def forward_trans(self, z, *args, **kwargs):
        if self.flow_forward:
            return self.training_direction(z, *args, **kwargs)
        return self.inverse_direction(z, *args, **kwargs)

    def backward_trans(self, x, *args, **kwargs):
        if self.flow_forward:
            return self.inverse_direction(x, *args, **kwargs)
        return self.inverse_direction(x, *args, **kwargs)
    
    def training_direction(self, z, *args, **kwargs):
        raise NotImplementedError()

    def inverse_direction(self, x, *args, **kwargs):
        raise NotImplementedError()

    def get_param_count(self):
        return self.parameter_count
