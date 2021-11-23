from .data import Dataset

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.bernoulli import Bernoulli


class LogisticRegression(Dataset):
    def __init__(
            self, 
            n_train=10000, 
            n_valid=1000, 
            n_test=1000, 
            dim=100,
            prior_low=-100,
            prior_high=100,
            rho=0.5):
        super().__init__()

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.dim = dim

        covariance_matrix = (1-rho) * torch.eye(dim) + rho * torch.ones(dim)
        multigaussian = MultivariateNormal(torch.zeros(dim), covariance_matrix)

        X_train = multigaussian.rsample((n_train,))
        X_valid = multigaussian.rsample((n_valid,))
        X_test = multigaussian.rsample((n_test,))
        beta = (prior_high - prior_low) * torch.rand((dim,1)) + prior_low

        y_train = Bernoulli(logits=X_train @ beta).rsample((n_train,))[:,None]
        y_valid = Bernoulli(logits=X_valid @ beta).rsample((n_valid,))[:,None]
        y_test = Bernoulli(logits=X_test @ beta).rsample((n_test,))[:,None]

        self.train_data = [X_train, y_train]
        self.valid_data = [X_valid, y_valid]
        self.test_data = [X_test, y_test]

    def log_prior(self, param):
        return 0

    def log_lik(self, param, data):
        X, y = data
        beta = param
        p = torch.sigmoid(beta @ data.T)

        log_l = y * torch.log(p) + (1-y) * torch.log(1-p)
        log_l = torch.sum(log_l, axis=1)

        return log_l


    
    
