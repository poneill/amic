"""Sample a boltzmann distribution with the gumbel max trick"""
import numpy as np
from project_utils import inverse_cdf_sample

def make_eps():
    eps = np.random.normal(size=5000000)
    return eps
    
def inverse_sampling(eps):
    ps = np.exp(-eps)/np.sum(-eps)
    sampler = inverse_cdf_sampler(ps)
    return sampler()

def gumbel_sampling(eps):
    return (eps + np.random.gumbel(size=len(eps))).argmax()
