from .data import Dataset

import torch
from torch.nn.functional import softplus
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma


class LinearRegression(Dataset):
    def __init__(self, n_train=10000, n_valid=1000, n_test=1000, dim=100):
        super().__init__()

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.dim = dim
        self.dim_input = dim + 1

        self.multigaussian = MultivariateNormal(torch.zeros(dim).to(self.device_), torch.eye(dim).to(self.device_))
        self.gammadistr = Gamma(torch.tensor(.5).to(self.device_), torch.tensor(.5).to(self.device_))

        X_train = self.multigaussian.rsample((n_train,))
        X_valid = self.multigaussian.rsample((n_valid,))
        X_test = self.multigaussian.rsample((n_test,))

        beta = self.multigaussian.rsample()
        sigma = self.gammadistr.rsample()

        self.y_gaussian = Normal(0, sigma)
        y_train = torch.einsum('np,p->n', X_train, beta) + self.y_gaussian.rsample((n_train,))
        y_valid = torch.einsum('np,p->n', X_valid, beta) + self.y_gaussian.rsample((n_valid,))
        y_test = torch.einsum('np,p->n', X_test, beta) + self.y_gaussian.rsample((n_test,))

        self.train_data = [X_train, y_train]
        self.valid_data = [X_valid, y_valid]
        self.test_data = [X_test, y_test]

    def log_prior(self, param):
        log_p = self.multigaussian.log_prob(param[:,:self.dim])

        log_p += self.gammadistr.log_prob(softplus(param[:, self.dim:])[:,0])

        return log_p.to(self.device)

    def log_lik(self, param, data):
        X, y = data
        beta = param[:, :self.dim]
        sigma = softplus(param[:, self.dim:])
        mean = beta @ X.T
        
        log_l = ((y - mean)**2)
        log_l = torch.sum(log_l , axis=1)[:, None]
        log_l = - sigma - 1/(X.size()[0]) * (1/(2*sigma**2)) * log_l

   
        return (log_l[:,0].to(self.device))


    
    
