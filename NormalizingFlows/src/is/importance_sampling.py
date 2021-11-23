import numpy as np
import rpy2.robjects
from rpy2.robjects import r
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectores import StrVector

if rpackages.isinstalled('loo'):
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=cranInd)
    utils.install_packages(StrVector('loo'))
loo = rpackages.importr('loo')

def importance_sampling(flow, var_model, S, statistics, mode='train'):
    if not isinstance(statistics, list):
        statistics = [statistics]

    samples, log_flow = flow.sample(S)
    samples = samples[-1]
    log_target = var_model.evaluate(samples, mode)

    weights = (log_target - log_flow).cpu().detach().numpy()
    weights = weights/np.sum(weights)
    weighted_samples = weights * samples.cpu().detach().numpy()

    return [stat(weighted_samples) for stat in statistics]

def psis(flow, var_model, S, statistics, mode='train', verbose=True):
    if not isinstance(statistics, list):
        statistics = [statistics]

    psis_ret, psis_k, samples = psis_diagnostic(flow, var_model, S, mode, verbose)
    psis_weights = np.array(list(psis_ret[0][0]))
    psis_weights = psis_weights/np.sum(psis_weights)

    weighted_samples = psis_weights * samples.cpu().detach().numpy()

    return [stat(weighted_samples) for stat in statistics]

def psis_diagnostic(flow, var_model, S, mode='train', verbose=True):
    samples, log_flow = flow.sample(S)
    samples = samples[-1]
    log_target = var_model.evaluate(samples, mode)

    log_diff = (log_target - log_flow).cpu().detach().numpy()
    log_vector = rpy2.robjects.FloatVector(list(log_diff))

    psis_ret = loo.psis(log_vector)
    psis_k = psis_ret[1][0][0]

    if verbose:
        if psis_k <= 0.5:
            print("PSIS estimated k is {}, and can safely use the flow. Using PSIS gives good convergence.".format(psis_k))
        elif psis_k <= 0.7:
            print("PSIS estimated k is {}, and further steps ought to be taken, yet a passable flow.".format(psis_k))
        else:
            print("PSIS estimated k is {}, and the flow model is not reliable as a approximation.".format(psis_k))

    return psis_ret, psis_k, samples

