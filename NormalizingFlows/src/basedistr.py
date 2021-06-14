import torch

class BaseDistribution:
    def __init__(self, dim_input, distr=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if distr is None:
            distr = torch.distributions.multivariate_normal.MultivariateNormal(
                            torch.zeros(dim_input).to(self.device),
                            torch.eye(dim_input).to(self.device))

        self.distr = distr
        self.dim_input = dim_input
        
    def sample(self, n):
        return self.distr.sample((n,)).to(self.device)

    def log_prob(self, x):
        return self.distr.log_prob(x.to(self.device)).to(self.device)

    def update_device(self, device):
        self.device = device
