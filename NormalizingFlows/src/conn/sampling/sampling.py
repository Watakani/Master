import numpy as np

class Sampling:
    def _create_masks(self, m, dim_net):
        masks = [np.ones((d1,d2)) for d1,d2 in zip(dim_net[1:], dim_net[:-1])]
        for ind, mask in enumerate(masks):
            for row in range(len(mask)):
                mask[row,:] *= [s.issubset(m[ind+1][row]) for s in m[ind]]

        
        return masks
