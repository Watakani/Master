from .data import Dataset

import torch
import torch.nn

class ToyDataset(Dataset):
    def __init__(self, samples=1000, validation_perc=0.1, test_perc=0.1, data_distr=None, dim_input=2, dtype=torch.float):
        super().__init__()

        if data_distr is None:
            data_distr = create_mvnormal(dim_input, dtype)

        self.dtype = dtype
        self.test_n = int(samples * test_perc)
        self.valid_n = int(samples * validation_perc)

        self.dim_input = dim_input
        self.n = samples
        self.train_n = self.n - self.valid_n - self.test_n


        self.train_data = data_distr.sample((self.train_n,))
        self.validation_data = data_distr.sample((self.valid_n,))
        self.test_data = data_distr.sample((self.test_n,))

        self.data_distr = data_distr

    def sample(self, n):
        return self.data_distr.sample((n,)).to(self.device)

    def evaluate(self, x):
        return self.data_distr.log_prob(x).to(self.device)


def create_mvnormal(dim_input, dtype):
    sigma = torch.ones((dim_input, dim_input), dtype=dtype) * 0.8
    sigma[range(dim_input), range(dim_input)] = 3.0
    mean = torch.rand(dim_input, dtype=dtype) * 8.0

    return torch.distributions.multivariate_normal.MultivariateNormal(
            mean, sigma)
