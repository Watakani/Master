from itertools import chain, combinations
from copy import deepcopy

from .samplingset_min import SamplingSetMin

class SamplingSetFull(SamplingSetMin):
    def __init__(self):
        super().__init__()

    def generate(self,c):
        self.sample_set = super().generate(c)
        new_set = set()
        for s in self.sample_set:
            for a in self._powerset(s):
                new_set |= set(map(frozenset, a))

        self.sample_set = deepcopy(new_set)
        return list(map(set, self.sample_set))

    def _powerset(self, s):
        l = list(s)
        return [combinations(l,r) for r in range(1,len(s)+1)]


