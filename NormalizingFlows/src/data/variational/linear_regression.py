from .data import Dataset

import torch
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

        self.multigaussian = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.gammadistr = Gamma(.5,.5)

        X_train = multigaussian.rsample((n_train,))
        X_valid = multigaussian.rsample((n_valid,))
        X_test = multigaussian.rsample((n_test,))

        beta = self.multigaussian.rsample().T
        sigma = self.gammadistr.rsample()

        self.y_gaussian = Normal(0, sigma)
        y_train = X_train @ beta + self.y_gaussian.rsample((n_train,)).reshape((n_train,1))
        y_valid = X_valid @ beta + self.y_gaussian.rsample((n_valid,)).reshape((n_valid,1))
        y_test = X_test @ beta + self.y_gaussian.rsample((n_test,)).reshape((n_test,1))

        self.train_data = [X_train, y_train]
        self.valid_data = [X_valid, y_valid]
        self.test_data = [X_test, y_test]

    def log_prior(self, param):
        log_p = self.multigaussian.log_prob(param[:,:self.dim])
        log_p += self.gamma.log_prob(param[:, self.dim:])
        return log_p

    def log_lik(self, param, data):
        X, y = data
        beta = param[:, :self.dim]
        sigma = param[:, self.dim:]
        mean = beta @ data.T
        log_l = torch.sum((y.T - mean)**2, axis=1)[:, None]
        log_l = -X.size()[0] * torch.log(sigma) - 1/(2*sigma**2) * log_l

        return log_l


    
    
