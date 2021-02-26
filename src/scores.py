import torch
import numpy as np

def log_likelihood(data, model_1, *add_models):
    with torch.no_grad():
        models = [model_1] + [model for model in add_models]
        log_likelihoods = []
        for model in models: 
            _, log_prob = model.evaluate(data)
            log_likelihoods.append(log_prob.detach().numpy())

    means = [np.mean(log_prob) for log_prob in log_likelihoods]
    return log_likelihoods, means

def difference_loglik(data, target, model_1, *add_models):
    with torch.no_grad():
        models = [model_1] + [model for model in add_models]
        diff_loglik = []
        target_loglik = target.evaluate(data)
        for model in models:
            _, log_prob = model.evaluate(data)
            diff_loglik.append(torch.abs(log_prob - target_loglik).detach().numpy())
 
    means = [np.mean(diff_log_prob) for diff_log_prob in diff_loglik]
    return diff_loglik, means
