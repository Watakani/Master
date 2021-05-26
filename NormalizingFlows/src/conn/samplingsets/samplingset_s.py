from copy import deepcopy

from .samplingset_min import SamplingSetMin

class SamplingSetS(SamplingSetMin):
    def __init__(self, cut_off=0):
        super().__init__()
        self.cut_off = cut_off

    def generate(self, c):
        self.sample_set = super().generate(c)
        new_set = deepcopy(self.sample_set)

        for s1 in self.sample_set:
            temp_set = deepcopy(new_set)
            for s2 in temp_set:
                intersect = set(s1) & set(s2)
                if len(intersect) > self.cut_off:
                    new_set |= {frozenset(intersect)}

        self.sample_set = deepcopy(new_set)
        return list(map(set, self.sample_set))
