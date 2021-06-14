import torch
import torch.nn as nn

from .conn import CONN
from .sampling.layerwise_sampling import LayerwiseSampling
from .samplingsets.samplingset_s import SamplingSetS

class MADE(CONN):
    def __init__(
            self, 
            dim_in, 
            dim_hidden, 
            dim_out, 
            act_func=nn.ReLU(),
            plural=1, 
            bias=True,
            sample_set_generator=SamplingSetS(),
            mask_sampling=LayerwiseSampling(),
            input_resid=True,
            output_resid=True):

        c = [set(list(range(i))) for i in range(dim_in)]

        super().__init__(dim_in, dim_hidden, dim_out, c, act_func, plural, bias,
                sample_set_generator, mask_sampling, input_resid, output_resid)

