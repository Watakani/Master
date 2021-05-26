class SamplingSetMin():
    def __init__(self):
        self.sample_set = set()

    def generate(self, c):
        self.sample_set = set(map(frozenset,c))
        return list(map(set, self.sample_set))

    def get_sample_set(self):
        return list(map(set, self.sample_set))
