from utils import invert_permutation

class IdendityPerm:
    def __init__(self, dim_input):
        self.permutation = list(range(dim_input))
        self.invert_permutation = invert_permutation(self.permutation)

    def permute(self):
        return self.permutation

    def inv_permute(self):
        return self.invert_permutation

def create_identity_perm(dim_input, num_trans):
    permutations = []

    for t in range(num_trans):
        permutations.append(IdendityPerm(dim_input))

    return permutations

