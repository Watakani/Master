from .data import Dataset

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.bernoulli import Bernoulli
import numpy as np


class LogisticRegression(Dataset):
    def __init__(
            self, 
            n_train=10000, 
            n_valid=1000, 
            n_test=1000, 
            dim=20,
            rho=0.7):
        super().__init__()

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.dim = dim
        self.dim_input = dim 

        covariance_matrix = (1-rho) * torch.eye(dim) + rho * torch.ones(dim)
        multigaussian = MultivariateNormal(torch.zeros(dim).to(self.device_), covariance_matrix.to(self.device_))

        X_train = multigaussian.rsample((n_train,))
        X_valid = multigaussian.rsample((n_valid,))
        X_test = multigaussian.rsample((n_test,))
        beta = torch.rand((dim,1)).to(self.device_)

        y_train = Bernoulli(logits=(X_train @ beta)[:,0]).sample()
        y_valid = Bernoulli(logits=(X_valid @ beta)[:,0]).sample()
        y_test = Bernoulli(logits=(X_test @ beta)[:,0]).sample()

    
        self.train_data = [X_train, y_train]
        self.valid_data = [X_valid, y_valid]
        self.test_data = [X_test, y_test]

    def log_prior(self, param):
        return 0

    def log_lik(self, param, data):
        X, y = data
        beta = param
        p = torch.clamp(torch.sigmoid(beta @ X.T),min=0.00000001, max=0.9999999)

        
        log_l = y * torch.log(p) + (1-y) * torch.log(1-p)
        log_l = torch.mean(log_l, axis=1)

        return log_l.to(self.device)


    
    
