import torch
import torch.nn as nn

class Transformation(nn.Module):
    def __init__(self, parameter_count):
        super.__init__()
        self.parameter_count

    def forward(self, x, param, forward=True):
        return self.forward_trans(x, param) if self.forward
                    else self.backward_trans(x, param)

    def forward_trans(self):
        pass

    def backward_trans(self):
        pass

    def get_param_count(self):
        return self.parameter_count
