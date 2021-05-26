import numpy as np

from .sampling import Sampling

class UniformSampling(Sampling):
    def __init__(self):
        super().__init__()

    def sample(self, c, sample_set, dim_net):
        sample_set_len = len(sample_set)
        m_0 = [{i} for i in range(dim_net[0])]
        #m_{L+1}, aka output layer
        m_L1 = c

        m_l = []
        for l in range(1,len(dim_net)-1):
            samples = [sample_set[i] for i in np.random.randint(sample_set_len, size=(dim_net[l]))]
            m_l.append(samples)

        m = [m_0] + m_l + [m_L1] 

        return self._create_masks(m, dim_net)
