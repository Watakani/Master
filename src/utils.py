def invert_permutation(permutation):
    inv_perm = [0] * len(permutation)

    for i in  range(len(permutation)):
        inv_perm[permutation[i]] = i
    return inv_perm

def permute_data(data, permutation):
    return data[:, permutation.permute()]

def inv_permute_data(data, permutation):
    return data[:, permutation.inv_permute()]

