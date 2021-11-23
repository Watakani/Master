import torch
import torch.nn as nn

def kl_backward(log_prob, mean=True):
    if mean:
        return -torch.mean(log_prob)
    return -torch.sum(log_prob)

def kl_forward(log_prob, log_target_prob, mean=True):
    if mean:
        return torch.mean(log_prob - log_target_prob)
    return torch.sum(log_prob - log_target_prob)

def chisq_div(log_prob, log_target_prob, df=2, mean=True):
    target_prob = torch.exp(log_target_prob)
    prob = torch.exp(log_prob)

    if mean:
        return torch.mean((target_prob/prob)**df)
    return torch.sum((target_prob/prob)**df)

def chisq_div_stable(log_prob, log_target_prob, df=2, mean=True):
    with torch.no_grad():
        log_w = log_target_prob - log_prob 
        w = torch.exp(log_w - torch.max(log_w))

    if mean:
        return torch.mean(w**df * log_prob)
    return torch.sum(w**df * log_prob)

