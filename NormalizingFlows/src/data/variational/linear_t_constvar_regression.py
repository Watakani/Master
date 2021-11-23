from .data import Dataset

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.studentT import StudentT


class LinearTRegression(Dataset):
    def __init__(self, n_train=10000, n_valid=1000, n_test=1000, dim=100, df=10):
        super().__init__()

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.dim = dim
        self.df = df

        x_covariance = 0.2 * torch.eye(dim) + 0.8 * torch.ones((dim,dim))
        self.x_distr = MultivariateNormal(torch.zeros(dim), x_covariance)
        self.beta_distr = MultivariateNormal(torch.zeros(dim), torch.ones(dim))

        X_train = self.x_distr.rsample((n_train,))
        X_valid = self.x_distr.rsample((n_valid,))
        X_test = self.x_distr.rsample((n_test,))

        beta = self.beta_distr.rsample().T

        y_train = StudentT(self.df, loc=X_train @ beta)
        y_valid = StudentT(self.df, loc=X_valid @ beta)
        y_test = StudentT(self.df, loc=X_test @ beta)

        self.train_data = [X_train, y_train]
        self.valid_data = [X_valid, y_valid]
        self.test_data = [X_test, y_test]

    def log_prior(self, param):
        log_p = self.beta_distr.log_prob(param)
        return log_p

    def log_lik(self, param, data):
        X, y = data
        beta = param
        mean = beta @ data.T
        log_l = torch.sum(torch.log(1 + 1/self.df * ((y.T - mean))**2), axis=1)[:, None]
        log_l = -(self.df + 1)/2 * log_l

        return log_l
