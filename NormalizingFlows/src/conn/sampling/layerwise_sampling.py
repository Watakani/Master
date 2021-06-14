import numpy as np 
from numpy.random import randint, choice

from .sampling import Sampling

class LayerwiseSampling(Sampling):
    def __init__(self):
        super().__init__()

    def sample(
            self, 
            c, 
            sample_set,
            dim_net, 
            alpha_1=.9,
            alpha_2=.9,
            delta=.1):


        sample_set_len = len(sample_set)
        history_length = int(np.log(delta)/np.log(alpha_2))
        history = np.zeros((len(dim_net)-2, sample_set_len))

        m_0 = [{i} for i in range(dim_net[0])]
        #m_{L+1}, aka output layer
        m_L1 = c

        m_l = []
        for l in range(1, len(dim_net)-1):
            if l==1:
                index_samples = randint(sample_set_len, size=(dim_net[l]))
                samples = [sample_set[i] for i in index_samples]

            elif l==2:
                p = np.exp(alpha_1 * sub_of)/np.sum(np.exp(alpha_1 * sub_of))
                index_samples = choice(np.arange(sample_set_len), size=(dim_net[l]), p=p[0,:])
                samples = [sample_set[i] for i in index_samples]

                history = history * alpha_2

            else:
                p = alpha_1 * sub_of
                p = p - np.sum(history[(max(0, l-history_length)):l-2, :], axis=0)
                p = np.exp(p)/np.sum(np.exp(p))

                index_samples = choice(np.arange(sample_set_len), size=(dim_net[l]), p=p[0,:])
                samples = [sample_set[i] for i in index_samples]

                history = history * alpha_2

            m_l.append(samples)

            sub_of = np.zeros((1, len(sample_set)))
            for ind, s_i in enumerate(sample_set):
                temp = np.sum([s.issubset(s_i) for s in m_l[-1]]) > 0
                sub_of[0,ind] = temp

            history[l-1,:] = [np.sum(index_samples == i) for i in range(sample_set_len)]

        
        m = [m_0] + m_l + [m_L1]

        return m, self._create_masks(m, dim_net)




