##Methods to create different known flows.

from permutations.alternate import create_alternate_perm
from permutations.random import create_random_perm
from permutations.identity import create_identity_perm
from structure.ar import AR
from structure.iar import IAR
from structure.twoblock import TwoBlock
from structure.identity import ID
from transforms.affine import Affine
from transforms.logit import Logit
from transforms.piecewise import PiecewiseAffine
from transforms.sigmoid import Sigmoid
from transforms.sigmoid_pos import SigmoidPos

from nf import NormalizingFlow

import torch
import torch.nn as nn

def get_permutation(perm_type, dim_input, num_trans):
    if perm_type == 'random':
        permutations = create_random_perm(dim_input, num_trans)
        
    elif perm_type == 'alternate':
        permutations = create_alternate_perm(dim_input, num_trans)

    else:
        permutations = create_identity_perm(dim_input, num_trans)

    return permutations

def create_std_gaussian(dim_input):
    return torch.distributions.multivariate_normal.MultivariateNormal(
                        torch.zeros(dim_input), 
                        torch.eye(dim_input))

def create_iaf(
        dim_input, 
        dim_hidden,
        num_trans, 
        perm_type='identity',
        kl_forward=True,
        base_distr=None,
        act_func=nn.ReLU()):

    if base_distr is None: 
        base_distr = create_std_gaussian(dim_input)

    permutations = get_permutation(perm_type, dim_input, num_trans)
    flow = []

    for t in range(num_trans):
        trans = Affine()
        struct = IAR(dim_input, dim_hidden, trans, permutations[t],
                        kl_forward, act_func)
        flow.append(struct)

    return NormalizingFlow(flow, base_distr, kl_forward)

def create_maf(
        dim_input,
        dim_hidden,
        num_trans,
        perm_type='identity',
        kl_forward=True,
        base_distr=None,
        act_func=nn.ReLU()):

    if base_distr is None: 
        base_distr = create_std_gaussian(dim_input)

    permutations = get_permutation(perm_type, dim_input, num_trans)
    flow = []

    for t in range(num_trans):
        trans = Affine()
        struct = AR(dim_input, dim_hidden, trans, permutations[t],
                        kl_forward, act_func)
        flow.append(struct)

    return NormalizingFlow(flow, base_distr, kl_forward)

def create_realnvp(
        dim_input,
        dim_hidden,
        num_trans,
        perm_type='identity',
        kl_forward=True,
        base_distr=None,
        act_func=nn.ReLU()):

    if base_distr is None: 
        base_distr = create_std_gaussian(dim_input)

    permutations = get_permutation(perm_type, dim_input, num_trans)
    flow = []

    for t in range(num_trans):
        trans = Affine()
        struct = TwoBlock(dim_input, dim_hidden, trans, permutations[t],
                            kl_forward, act_func)
        flow.append(struct)

    return NormalizingFlow(flow, base_distr, kl_forward)

def create_paf(
        dim_input,
        dim_hidden,
        num_trans,
        perm_type='identity',
        kl_forward=True,
        base_distr=None,
        act_func=nn.ReLU()):

    if base_distr is None: 
        base_distr = create_std_gaussian(dim_input)
    permutations = get_permutation(perm_type, dim_input, num_trans)
    flow = []

    for t in range(num_trans):
        trans = PiecewiseAffine()
        struct = IAR(dim_input, dim_hidden, trans, permutations[t], 
                        kl_forward, act_func)
        flow.append(struct)

    return NormalizingFlow(flow, base_distr, kl_forward)

def create_flows(
        dim_input,
        dim_hidden,
        num_trans,
        perm_type='identity',
        kl_forward=True,
        base_distr=None,
        structure=IAR,
        transformation=PiecewiseAffine,
        act_func=nn.ReLU()):

    if base_distr is None: 
        base_distr = create_std_gaussian(dim_input)
    permutations = get_permutation(perm_type, dim_input, num_trans)
    flow = []

    if not isinstance(transformation, list):
        transformation = [transformation] * num_trans

    if not isinstance(structure, list):
        structure = [structure] * num_trans

    for t in range(num_trans):
        trans = transformation[t]()
        struct = structure[t](dim_input, dim_hidden, trans, permutations[t], 
                        kl_forward, act_func)
        flow.append(struct)

    return NormalizingFlow(flow, base_distr, kl_forward)

