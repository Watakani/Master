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
        x, log_det = self.transform(z, forward=True)
        return x, log_det

    def backward_flow(self, x):
        z, log_det = self.transform(z, forward=False)
        return z, log_det

    def update_device(self, device):
        "Does not have use device"
        pass
