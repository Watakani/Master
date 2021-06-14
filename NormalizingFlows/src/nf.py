import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class NormalizingFlow(nn.Module):
    def __init__(
            self, 
            flow,
            base_distr,
            flow_forward=True,
            name='flow_model'):

        super().__init__()

        self.flow = nn.ModuleList(flow)
        self.flow_forward = flow_forward

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.base_distr = base_distr
        self.name = name

    def forward(self, x):
        return self.forward_flow(x) if self.flow_forward else self.backward_flow(x)

    def forward_flow(self, z_0):
        dim_row, _ = x.shape
        log_prob = torch.zeros(dim_row, device=self.device)
        log_prob += self.base_distr.log_prob(z_0)
        z = [z_0]
        z_i = z_0

        for f in self.flow:
            z_i, log_det_i = f(z_i)
            log_prob -= log_det_i
            z.append(z_i)

        return z, log_prob

    def backward_flow(self, x):
        dim_row, _ = x.shape
        log_prob = torch.zeros(dim_row, device=self.device)
        z = [x] 
        z_i = x

        for f in self.flow[::-1]:
            z_i, log_det_i = f(z_i)
            log_prob += log_det_i
            z.append(z_i)

        log_prob += self.base_distr.log_prob(z[-1])
        return z[::-1], log_prob

    def sample(self, n):
        z_0 = self.base_distr.sample(n).to(self.device)
        log_prob = torch.zeros(n, device=self.device)
        log_prob += self.base_distr.log_prob(z_0)

        z = [z_0]
        z_i = z_0

        for f in self.flow:
            z_i, log_det_i = f.forward_flow(z_i)
            log_prob -= log_det_i
            z.append(z_i)

        return z, log_prob
    
    def evaluate(self, x):
        dim_row, _ = x.shape
        log_prob = torch.zeros(dim_row, device=self.device)
        z = [x]
        z_i = x

        for f in self.flow[::-1]:
            z_i, log_det_i = f.backward_flow(z_i)
            log_prob += log_det_i
            z.append(z_i)

         
        log_prob += self.base_distr.log_prob(z[-1])
        return z[::-1], log_prob

    def get_base_distr(self):
        return self.base_distr

    def update_device(self, device):
        self.device = device
        for f in self.flow:
            f.update_device(device)
        self.base_distr.update_device(device)

    def freeze_trans(self, trans_to_freeze, exclude=False):
        if isinstance(trans_to_freeze, int):
            if exclude:
                temp = list(range(len(self.flow)))
                temp.remove(trans_to_freeze)
                trans_to_freeze = temp

            else:
                trans_to_freeze = list(range(trans_to_freeze))

        for i in trans_to_freeze:
            self.flow[i].requires_grad_(False)

    def unfreeze_trans(self, trans_to_unfreeze=None):
        if trans_to_unfreeze is None:
            trans_to_unfreeze = list(range(len(self.flow)))

        elif isinstance(trans_to_unfreeze, int):
            trans_to_unfreeze = list(range(trans_to_unfreeze))

        for i in trans_to_unfreeze:
            self.flow[i].requires_grad_(True)

    def __str__(self):
        return self.name
