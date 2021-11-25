##Methods to create different known flows.

from .permutations.alternate import create_alternate_perm
from .permutations.random import create_random_perm
from .permutations.identity import create_identity_perm

from .structure.ar import AR
from .structure.iar import IAR
from .structure.full import Full
from .structure.twoblock import TwoBlock
from .structure.identity import ID

from .transforms.affine import Affine
from .transforms.piecewise import PiecewiseAffine
from .transforms.piecewise_affine import PiecewiseAffineAffine
from .transforms.continuous_piecewise import ContinuousPiecewise
from .transforms.continuous_piecewise_affine import ContinuousPiecewiseAffineAffine
from .transforms.linear_trans import LinearTrans
from .transforms.constant import Constant

from .nf import NormalizingFlow
from .basedistr import BaseDistribution

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_permutation(perm_type, dim_input, num_trans):
    if perm_type == 'random':
        permutations = create_random_perm(dim_input, num_trans)
        
    elif perm_type == 'alternate':
        permutations = create_alternate_perm(dim_input, num_trans)

    else:
        permutations = create_identity_perm(dim_input, num_trans)

    return permutations

def create_flows_with_IAR(
        dim_input,
        dim_hidden,
        transformations,
        perm_type='identity',
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    permutations = get_permutation(perm_type, dim_input, len(transformations))
    flow = []

    for t, trans in enumerate(transformations):
        struct = IAR(dim_input, dim_hidden, trans, permutations[t], 
                    flow_forward, act_func, *args, **kwargs)

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_AR(
        dim_input,
        dim_hidden,
        transformations,
        perm_type='identity',
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    permutations = get_permutation(perm_type, dim_input, len(transformations))
    flow = []

    for t, trans in enumerate(transformations):
        struct = AR(dim_input, dim_hidden, trans, permutations[t], 
                    flow_forward, act_func, *args, **kwargs)

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_twoblock(
        dim_input,
        dim_hidden,
        transformations,
        perm_type='identity',
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    permutations = get_permutation(perm_type, dim_input, len(transformations))
    flow = []

    for t, trans in enumerate(transformations):
        struct = TwoBlock(dim_input, dim_hidden, trans, permutations[t], 
                    flow_forward, act_func, *args, **kwargs)

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_alt_identity_twoblock(
        dim_input,
        dim_hidden,
        transformations,
        perm_type='identity',
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    permutations = get_permutation(perm_type, dim_input, len(transformations))
    flow = []
    perm_i = 0
    for t, trans in enumerate(transformations):
        if t % 2 == 0:
            struct = ID(dim_input, trans, flow_forward, 
                act_func, *args, **kwargs)
        else:
            struct = TwoBlock(dim_input, dim_hidden, trans, permutations[perm_i], 
                        flow_forward, act_func, *args, **kwargs)
            perm_i += 1

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_alt_identity_AR(
        dim_input,
        dim_hidden,
        transformations,
        perm_type='identity',
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    permutations = get_permutation(perm_type, dim_input, len(transformations))
    flow = []
    
    perm_i = 0
    for t, trans in enumerate(transformations):
        if t % 2 == 0:
            struct = ID(dim_input, trans, flow_forward, 
                act_func, *args, **kwargs)
        else:
            struct = AR(dim_input, dim_hidden, trans, permutations[perm_i], 
                    flow_forward, act_func, *args, **kwargs)
            perm_i += 1

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_alt_identity_IAR(
        dim_input,
        dim_hidden,
        transformations,
        perm_type='identity',
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    permutations = get_permutation(perm_type, dim_input, len(transformations))
    flow = []
    
    perm_i = 0
    for t, trans in enumerate(transformations):
        if t % 2 == 0:
            struct = ID(dim_input, trans, flow_forward, 
                act_func, *args, **kwargs)            
        else:
            struct = IAR(dim_input, dim_hidden, trans, permutations[perm_i], 
                    flow_forward, act_func, *args, **kwargs)
            perm_i += 1

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_full(
        dim_input,
        dim_hidden,
        transformations,
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    flow = []

    for t, trans in enumerate(transformations):
        struct = Full(dim_input, dim_hidden, trans,
                    flow_forward, act_func, *args, **kwargs)

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)

def create_flows_with_identity(
        dim_input,
        transformations,
        flow_forward=True,
        base_distr=None,
        act_func=nn.ReLU(),
        *args,
        **kwargs):

    if base_distr is None: 
        base_distr = BaseDistribution(dim_input)

    flow = []

    for t, trans in enumerate(transformations):
        struct = ID(dim_input, trans, flow_forward, 
                act_func, *args, **kwargs)

        flow.append(struct)

    return NormalizingFlow(flow, base_distr, flow_forward)


def create_affine_trans(
        num_trans,
        flow_forward=True,
        a_param=F.softplus):

    return [Affine(flow_forward, a_param) for t in range(num_trans)] 

def create_constant_trans(
        num_trans,
        dim_in,
        flow_forward=True,
        a_param=F.softplus):

    return [Constant(dim_in, flow_forward, a_param) for t in range(num_trans)]

def create_continuous_piecewise_trans(
        num_trans,
        forward_flow=True,
        beta_as_hyper=True,
        beta=5,
        a_param=torch.abs):

    return [ContinuousPiecewise(forward_flow, beta_as_hyper, beta, a_param)
                for t in range(num_trans)]

def create_piecewise_trans(
        num_trans,
        forward_flow=True,
        a_param=F.softplus):

    return [PiecewiseAffine(forward_flow, a_param) for t in range(num_trans)]

def create_alt_piecewise_affine_trans(
        num_trans,
        forward_flow=True,
        a_param=F.softplus,
        c_param=F.softplus):
    
    transforms = []
    for t in range(num_trans):
        if t % 2 == 0:
            transforms.append(PiecewiseAffine(forward_flow, a_param))

        else:
            transforms.append(Affine(forward_flow, c_param))

    return transforms

def create_affinepiecewise_trans(
        num_trans,
        forward_flow=True,
        a_param=F.softplus,
        c_param=F.softplus):

    return [PiecewiseAffineAffine(forward_flow, a_param, c_param) 
                for t in range(num_trans)]

def create_affinecontinuous_trans(
        num_trans,
        forward_flow=True,
        a_param=torch.abs,
        c_param=F.softplus,
        beta_as_hyper=True,
        beta=5
        ):
    return [ContinuousPiecewiseAffineAffine(forward_flow, a_param, c_param,
            beta_as_hyper, beta) for t in range(num_trans)]

def create_linear_bias_trans(
        num_trans,
        dim_in,
        forward_flow=True):

    return [LinearTrans(dim_in, forward_flow) for t in range(num_trans)]

def create_alt_linear_affine_trans(
        num_trans,
        dim_in,
        forward_flow=True,
        a_param=F.softplus):

    transforms = []
    for t in range(num_trans):
        if t%2==0:
            transforms.append(LinearTrans(dim_in, forward_flow))
        else:
            transforms.append(Affine(forward_flow, a_param))
    return transforms

def create_alt_linear_piecewise_trans(
        num_trans,
        dim_in,
        forward_flow=True,
        a_param=F.softplus):

    transforms = []
    for t in range(num_trans):
        if t%2==0:
            transforms.append(LinearTrans(dim_in, forward_flow))
        else:
            transforms.append(PiecewiseAffine(forward_flow, a_param))
    return transforms

def create_alt_linear_continuous_trans(
        num_trans,
        dim_in,
        forward_flow=True,
        beta_as_hyper=True,
        beta=5,
        a_param=torch.abs):

    transforms = []
    for t in range(num_trans):
        if t%2==0:
            transforms.append(LinearTrans(dim_in, forward_flow))
        else:
            transforms.append(ContinuousPiecewise(forward_flow, beta_as_hyper,
                    beta, a_param))

    return transforms

def create_alt_linear_affinepiecewise_trans(
        num_trans,
        dim_in,
        forward_flow=True,
        a_param=F.softplus,
        c_param=F.softplus):

    transforms = []
    for t in range(num_trans):
        if t%2==0:
            transforms.append(LinearTrans(dim_in, forward_flow))
        else:
            transforms.append(PiecewiseAffineAffine(forward_flow, a_param, c_param))

    return transforms

def create_alt_linear_affinecontinuous_trans(
        num_trans,
        dim_in,
        forward_flow=True,
        a_param=torch.abs,
        c_param=F.softplus,
        beta_as_hyper=True,
        beta=5):

    transforms = []
    for t in range(num_trans):
        if t%2==0:
            transforms.append(LinearTrans(dim_in, forward_flow))
        else:
            transforms.append(
                    ContinuousPiecewiseAffineAffine(forward_flow, a_param, 
                    c_param, beta_as_hyper, beta))

    return transforms
