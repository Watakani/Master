import torch
import torch.nn as nn


class ID(nn.Module):
    def __init__(
            self,
            transform,
            kl_forward=True):

        super().__init__()

        self.transform = transform
        self.kl_forward = kl_forward

    def forward(self, x):
        return self.backward_flow(x) if self.kl_forward else self.forward_flow(x)

    def forward_flow(self, z):
        return self.transform(z, forward=True)

    def backward_flow(self, x):
        return self.transform(x, forward=False)
