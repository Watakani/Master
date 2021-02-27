from data.data import Dataset

import torch
import torch.nn

class ToyDataset(Dataset):
    def __init__(self, samples=1000, num_train=0.8, data_distr=None, dim_input=2):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if data_distr is None:
            data_distr = create_mvnormal(dim_input)
        
        if num_train < 1:
            num_train = int(samples * num_train)
        
        self.train_data = data_distr.sample((num_train,)).to(device)
        self.test_data = data_distr.sample((samples - num_train,)).to(device)
        self.data_distr = data_distr

    def get_training_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def sample(self, n):
        return self.data_distr.sample((n,)).to(device)

    def evaluate(self, x):
        return self.data_distr.log_prob(x).to(device)


def create_mvnormal(dim_input):
    sigma = torch.ones((dim_input, dim_input)) * 0.8
    sigma[range(dim_input), range(dim_input)] = 1.0
    mean = torch.rand(dim_input) * 8.0

    return torch.distributions.multivariate_normal.MultivariateNormal(
                        mean, sigma)
