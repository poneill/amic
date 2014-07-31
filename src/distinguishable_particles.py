"""Tue Jul 15 14:12:51 EDT 2014

So far we have been assuming distinguishable particles.  In many
sources (e.g. Phillips et al.'s PBOTC) however, the statistics of
chromosomal configuration are derived using indistinguishable
particles.  The purpose of this script is to explore the differences
between these two assumptions.

"""
import itertools
from utils import fac,choose

G = 10
q = 5

def hamil(ss,eps):
    return [ep*s for s,ep in zip(ss,eps)]

def chi_from_tau(tau_config,G):
    """Convert a tau-config to a chi-config"""
    chi_config = [0] * G
    for t in tau_config:
        chi_config[t] = 1
    return chi_config
    
def indist_configs(G,q):
    """Return all ch-configs, assuming indistinguishable particles"""
    return [chi_from_tau(tau_config,G) for qp in range(q+1)
            for tau_config in itertools.combinations(range(G),qp)]
            #for degeneracy in range(int(fac(sum(tau_config))*choose(q,sum(tau_config))))]

def dist_configs(G,q):
    """Return all ch-configs, assuming indistinguishable particles"""
    return [chi_from_tau(tau_config,G) for qp in range(q+1)
            for tau_config in itertools.combinations(range(G),qp)
            for degeneracy in range(int(fac(q)/fac(show(q-qp))))]

def dist_Z(G,q,eps):
    return sum(hamil(ss,eps) for ss in foo)
