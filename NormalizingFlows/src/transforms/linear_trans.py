#Rewritten from https://github.com/PolterZeit/invertible_encoders

import torch
from torch import nn
from torch.nn import functional as F

from .transformation import Transformation

def haar_orthogonal(weight : torch.Tensor):
    assert weight.dim() == 2, f'Dimension must be 2, got {weight.dim()}.'
    m, n = weight.shape
    with torch.no_grad():
        x = torch.randn_like(weight)

        if m < n:
            x = x.T

        Q, R = torch.linalg.qr(x)
        d = torch.diag(R).sign()
        Q *= d.unsqueeze(-2).expand_as(Q)

        if m < n:
            Q = Q.T

        if m == n:
            mask = (torch.det(Q) > 0.0).float()
            mask[mask == 0.0] = -1.0
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(Q)
            Q[..., 0] *= mask[..., 0]

    return Q

def haar_orthogonal_(weight : torch.Tensor):
    weight.copy_(haar_orthogonal(weight))
    return weight

class LinearTrans(Transformation):
    weight : torch.Tensor

    def __init__(self, dim, forward_flow=True, type = 'lie', bias=True):
        super().__init__(dim*dim, forward_flow)
        self.dim = dim
        self._weight = nn.Parameter(torch.zeros(dim, dim))
        self.register_buffer(
            '_base',
            torch.zeros(dim, dim)
        )
        self.initialize_base()
        assert type in ['lie', 'cayley'], f'Type {type} not supported.'
        self.type = type

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.rand(1, dim))
            

    @property
    def weight(self) -> torch.Tensor: 
        M = self._weight.tril(-1)
        M = M - M.T
        if self.type == 'lie':
            return self._base @ torch.matrix_exp(M)

        elif self.type == 'cayley':
            I = torch.eye(self.features, device=self._weight.device)
            return self._base @ (I - M) @ torch.linalg.inv(I + M)

    @weight.setter
    def weight(self, value) -> torch.Tensor:
        with torch.no_grad():
            self._base.copy(value)
            self._weight = torch.zeros_like(value)

    def initialize_base(self):
        haar_orthogonal_(self._base)

    def extra_repr(self): 
        return 'features={}, type={}'.format(self.dim, self.type)

    def training_direction(self, z):
        log_det = torch.sum(torch.zeros_like(z), dim=1)
        if self.bias is not None:
            return F.linear(z, self.weight)+self.bias, log_det
        return F.linear(z, self.weight), log_det

    def inverse_direction(self, x):
        log_det = torch.sum(torch.zeros_like(x), dim=1)
        if self.bias is not None:
            return F.linear(x-self.bias, self.weight.T), log_det
        return F.linear(x, self.weight.T), log_det
