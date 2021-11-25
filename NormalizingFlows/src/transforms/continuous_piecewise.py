import torch
import torch.nn as nn

from .transformation import Transformation



def g_p(x, beta):
    log_det = x/beta
    return beta * (torch.exp(x/beta)-1), log_det

def g_n(x, beta):
    log_det = torch.log(beta/(beta + torch.abs(x)))
    return beta * torch.log((torch.abs(x)/beta) + 1), log_det

def aff_p(x, a, c_1, c_2):
    log_det = torch.log(a)
    return a * (x-c_1) + c_2, log_det

def aff_n(x, a, c_1, c_2):
    log_det = -torch.log(a)
    return ((x-c_2)/a) + c_1, log_det

class ContinuousPiecewise(Transformation):
    def __init__(self, 
            forward_flow=True, 
            beta_as_hyper=True, 
            beta=5, 
            a_param=torch.abs):

        self.beta_as_hyper = beta_as_hyper
        self.a_param = a_param
        self.zero = None

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
        a_p = self.a_param(a)/2.0 + 1

        c_1 = self.beta * torch.log(a_p) 
        c_2 = self.beta * (a_p - 1)
        
        if self.zero is None:
            self.zero = torch.zeros_like(z)

        
        i_1 = a > 0
        i_2 = a <= 0
        i_3 = z > c_1
        i_4 = (z >= 0) & (z <= c_1)
        i_5 = z > c_2
        i_6 = (z >= 0) & (z <= c_2)

        x_1 , log_det_1 = (g_p(z, self.beta))        
        x_2, log_det_2 = g_n(z, self.beta)        
        x_3, log_det_3 = aff_p(z, a_p, c_1, c_2)        
        x_4, log_det_4 = aff_n(z, a_p, c_1, c_2)

    
        x = x_1 * (i_1 & i_4) + x_2 * (i_2 & i_6) + x_3 * (i_1 & i_3) + x_4 * (i_2 & i_5) + torch.min(z, self.zero)
        log_det = log_det_1 * (i_1 & i_4) + log_det_2 * (i_2 & i_6) + log_det_3 * (i_1 & i_3) + log_det_4 * (i_2 & i_5)
        x += b

        return x, torch.sum(log_det, dim=1)

    def inverse_direction(self, x, param, *args, **kwargs):
        if self.beta_as_hyper:
            a,b = param[0], param[1]

        else:
            a,b, self.beta = param[0], param[1], F.softplus(param[2])

        x = x - b
        a_p = self.a_param(a)/2.0 + 1

        c_1 = self.beta * torch.log(a_p) 
        c_2 = self.beta * (a_p - 1)
        
        if self.zero is None:
            self.zero = torch.zeros_like(z)

        
        i_1 = a <= 0
        i_2 = a >  0
        i_3 = z > c_1
        i_4 = (z >= 0) & (z <= c_1)
        i_5 = z > c_2
        i_6 = (z >= 0) & (z <= c_2)

        z_1, log_det_1 = (g_p(x, self.beta))        
        z_2, log_det_2 = g_n(x, self.beta)        
        z_3, log_det_3 = aff_p(x, a_p, c_1, c_2)        
        z_4, log_det_4 = aff_n(x, a_p, c_1, c_2)

    
        z = z_1 * (i_1 & i_4) + z_2 * (i_2 & i_6) + z_3 * (i_1 & i_3) + z_4 * (i_2 & i_5) + torch.min(x, self.zero)
        log_det = log_det_1 * (i_1 & i_4) + log_det_2 * (i_2 & i_6) + log_det_3 * (i_1 & i_3) + log_det_4 * (i_2 & i_5)
        
        z += b

        return x, torch.sum(log_det, dim=1)
