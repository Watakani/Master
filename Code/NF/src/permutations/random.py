from src.utils import invert_permutation
import numpy as np

class RandomPerm:
    def __init__(self, dim_input):
        self.permutation = np.random.permutation(dim_input).tolist()
        self.inv_permutation = invert_permutation(self.permutation)

    def permute(self):
        return self.permutation

    def inv_permute(self):
        return self.inv_permutation

def create_random_perm(dim_input, num_trans):
    permutations = []

    for d in range(dim_input):
        permutations.append(RandomPerm(dim_input))
    return permutations
