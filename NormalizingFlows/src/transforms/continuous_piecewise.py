import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformation import Transformation


def g_p(x, beta):
    log_det = x/beta
    return beta * (torch.exp(x/beta)-1), log_det

def g_n(x, beta):
    log_det = torch.log(beta/(beta + x))
    return beta * torch.log((x/beta) + 1), log_det

def aff_p(x, a, c_1, c_2):
    log_det = torch.log(a)
    return a * (x-c_1) + c_2, log_det

def aff_n(x, a, c_1, c_2):
    log_det = -torch.log(a)
    return ((x-c_2)/a) + c_1, log_det

def h_p(z, a, c_1, c_2, beta):
    x = z.clone()
    log_det = torch.zeros_like(z)
    i_1 = z > c_1
    i_2 = (z >= 0) & (z <= c_1)

    x[i_1], log_det_1 = aff_p(z[i_1], a[i_1], c_1[i_1], c_2[i_1])
    x[i_2], log_det_2 = g_p(z[i_2], beta)

    log_det[i_1] += log_det_1
    log_det[i_2] += log_det_2

    return x, log_det

def h_n(z, a, c_1, c_2, beta):
    x = z.clone()
    log_det = torch.zeros_like(z)
    i_1 = z > c_2
    i_2 = (z >= 0) & (z <= c_2)

    x[i_1], log_det_1 = aff_n(z[i_1], a[i_1], c_1[i_1], c_2[i_1])
    x[i_2], log_det_2 = g_n(z[i_2], beta)

    log_det[i_1] += log_det_1
    log_det[i_2] += log_det_2

    return x, log_det

class ContinuousPiecewise(Transformation):
    def __init__(self, 
            forward_flow=True, 
            beta_as_hyper=True, 
            beta=5, 
            a_param=torch.abs):

        self.beta_as_hyper = beta_as_hyper
        self.a_param = a_param

        if beta_as_hyper:
            super().__init__(2, forward_flow)
            self.beta = beta

        else:
            super().__init__(3, forward_flow)

    def training_direction(self, z, param, *args, **kwargs):
        if self.beta_as_hyper:
            a,b = param[0], param[1]

        else:
            a,b, self.beta = param[0], param[1], F.softplus(param[2])

        z = z - b
        a_p = self.a_param(a) + 1

        c_1 = self.beta * torch.log(a_p) 
        c_2 = self.beta * (a_p - 1)

        x = z.clone()
        log_det = torch.zeros_like(z)

        i_1 = a > 0
        i_2 = a <= 0

        x[i_1], log_det_1 = h_p(z[i_1], a_p[i_1], c_1[i_1], c_2[i_1], self.beta)
        x[i_2], log_det_2 = h_n(z[i_2], a_p[i_2], c_1[i_2], c_2[i_2], self.beta)

        log_det[i_1] += log_det_1
        log_det[i_2] += log_det_2

        x += b

        return x, torch.sum(log_det, dim=1)

    def inverse_training(self, x, param, *args, **kwargs):
        if self.beta_as_hyper:
            a,b = param[0], param[1]

        else:
            a,b, self.beta = param[0], param[1], F.softplus(param[2])

        x = x - b
        a_p = self.a_param(a) + 1

        c_1 = self.beta * torch.log(a_p) 
        c_2 = self.beta * (a_p - 1)

        z = x.clone()
        log_det = torch.zeros_like(x)

        i_1 = a <= 0
        i_2 = a > 0

        z[i_1], log_det_1 = h_p(x[i_1], a_p[i_1], c_1[i_1], c_2[i_1], self.beta)
        z[i_2], log_det_2 = h_n(x[i_2], a_p[i_2], c_1[i_2], c_2[i_2], self.beta)

        log_det[i_1] += log_det_1
        log_det[i_2] += log_det_2

        z += b

        return z, torch.sum(log_det, dim=1)
