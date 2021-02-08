from src.utils import invert_permutation

class IdendityPerm:
    def __init__(self, dim_input):
        self.permutation = list(range(dim_input))
        self.invert_permutation = invert_permutation(self.permutation)

    def permute(self):
        return self.permutation

    def inv_permute(self):
        return self.inv_permute

def create_identity_perm(dim_input, num_trans):
    permutations = []

    for d in range(dim_input):
        permutations.append(AlternatePerm(dim_input))

    return permutations

