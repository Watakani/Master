import torch
import torch.nn as nn

from .transformation import Transformation
from .affine import Affine
from .continuous_piecewise import ContinuousPiecewise

class ContinuousPiecewiseAffineAffine(Transformation):
    def __init__(self, 
            forward_flow=True, 
            a_param=torch.abs, 
            c_param=torch.exp,
            beta_as_hyper=True,
            beta=5):

        super().__init__(4, forward_flow)
        self.cont_piecewise =ContinuousPiecewise(forward_flow, a_param=a_param,
                    beta_as_hyper=beta_as_hyper, beta=beta)
        self.affine = Affine(forward_flow=forward_flow, a_param=c_param)

    def training_direction(self, z, param):
        a, b, c, d = param[0], param[1], param[2], param[3]


        x, log_det_1 = self.cont_piecewise.training_direction(z, [a,b])
        
        x, log_det_2 = self.affine.training_direction(x, [c,d])

        return x, (log_det_1 + log_det_2)

    def inverse_direction(self, x, param):
        a, b, c, d = param[0], param[1], param[2], param[3]

        z, log_det_1 = self.affine.inverse_direction(x, [c,d])

        z, log_det_2 = self.cont_piecewise.inverse_direction(z, [a,b])

        return z, (log_det_1 + log_det_2)
