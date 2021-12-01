from .data import Dataset

import torch
from torch.nn.functional import softplus
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
        self.dim_input = dim+1
        self.df = df

        x_covariance = 0.2 * torch.eye(dim) + 0.8 * torch.ones((dim,dim))
        self.x_distr = MultivariateNormal(torch.zeros(dim).to(self.device_), x_covariance.to(self.device_))
        self.beta_distr = MultivariateNormal(torch.zeros(dim).to(self.device_), torch.eye(dim).to(self.device_))
        self.gammadistr = Gamma(torch.tensor(.5).to(self.device_), torch.tensor(.5).to(self.device_))

        X_train = self.x_distr.rsample((n_train,))
        X_valid = self.x_distr.rsample((n_valid,))
        X_test = self.x_distr.rsample((n_test,))

        beta = self.beta_distr.rsample().T
        sigma = self.gammadistr.rsample()

        y_train = StudentT(self.df, loc=X_train @ beta, scale=sigma).sample()
        y_valid = StudentT(self.df, loc=X_valid @ beta, scale=sigma).sample()
        y_test = StudentT(self.df, loc=X_test @ beta, scale=sigma).sample()

        self.train_data = [X_train, y_train]
        self.valid_data = [X_valid, y_valid]
        self.test_data = [X_test, y_test]

    def log_prior(self, param):
        log_p = self.beta_distr.log_prob(param[:,:self.dim])
        log_p += self.gammadistr.log_prob(softplus(param[:, self.dim:])[:,0])

        return log_p.to(self.device)

    def log_lik(self, param, data):
        X, y = data
        beta = param[:, :self.dim]
        sigma = softplus(param[:, self.dim:])
        mean = beta @ X.T

        a = (y-mean)**2
        b = a/(sigma**2)
        c = 1/self.df * b
        d = torch.log(1 + c)
        
        log_l = torch.sum(d, axis=1)[:, None]
        log_l = - torch.log(sigma) - (self.df + 1)/2 * log_l

        return ((1/(X.size()[0]) * log_l)[:,0]).to(self.device)
    
