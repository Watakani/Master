import torch
import torch.nn as nn

def BatchNorm1D(nn.Module):
    def __init__(self, 
            num_features, 
            eps=1e-05, 
            momentum=0.1,
            affine=True):

        super().__init__()
        self.affine = affine
        self.eps = eps
        self.batchnorm = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        y, log_det = self.forward_norm(x)
        self.batchnorm(x)
        return y, log_det

    def forward_norm(self, x):
        std = torch.sqrt(self.batchnorm.running_var + self.eps)
        log_det = -torch.log(torch.ones_like(x) * std)

        y = x - self.batchnorm.running_mean
        y = y/std

        if self.affine:
            y = y * self.batchnorm.weight + self.batchnorm.bias
            log_det = log_det + torch.log(self.batchnorm.weight)

        log_det = torch.sum(log_det, axis=1)

        return y, log_det

    def backward_norm(self, x):
        y = x

        if self.affine:
            y = (y - self.batchnorm.bias)/self.batchnorm.weight

        std = torch.sqrt(self.batchnorm.running_var + eps)
        y = y * std + self.batchnorm.running_mean

        log_det = torch.log(std * torch.ones_like(x))
        if self.affine: log_det = log_det - torch.log(self.batchnorm.weight)
        log_det = torch.sum(log_det, axis=1)

        return y, log_det

            
