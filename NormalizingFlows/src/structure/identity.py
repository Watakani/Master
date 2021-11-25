import torch
import torch.nn as nn

from ..transforms.constant import Constant

class ID(nn.Module):
    def __init__(
            self,
            dim_in, 
            transform,
            flow_forward=True,
            *args,
            **kwargs):

        super().__init__()

        self.transform = transform
        self.flow_forward = flow_forward

    def forward(self, x):
        return self.forward_flow(x) if self.flow_forward else self.backward_flow(x)

    def forward_flow(self, z):
        x, log_det = self.transform(z, forward=True)
        return x, log_det

    def backward_flow(self, x):
        z, log_det = self.transform(x, forward=False)
        return z, log_det

    def update_device(self, device):
        "Does not have use device"
        pass
