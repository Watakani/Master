import torch
import torch.nn as nn

class Transformation(nn.Module):
    def __init__(self, parameter_count):
        super().__init__()
        self.parameter_count = parameter_count

    def forward(self, *x, forward=True):
        return self.forward_trans(*x) if forward else self.backward_trans(*x)

    def forward_trans(self):
        raise NotImplementedError()

    def backward_trans(self):
        raise NotImplementedError()

    def get_param_count(self):
        return self.parameter_count
