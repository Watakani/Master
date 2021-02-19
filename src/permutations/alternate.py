from utils import invert_permutation

class AlternatePerm:
    def __init__(self, dim_input, even):
        self.permutation = list(range(dim_input))
        if not even:
            self.permutation = self.permutation[::-1]

        self.inv_permutation = invert_permutation(self.permutation)

    def permute(self):
        return self.permutation
    
    def inv_permute(self):
        return self.inv_permutation

def create_alternate_perm(dim_input, num_trans):
    permutations = []

    for t in range(num_trans):
        permutations.append(AlternatePerm(dim_input, t%2==0))

    return permutations
