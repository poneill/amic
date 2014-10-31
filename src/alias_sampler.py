"""
Implement Vose's method for alias sampling

"""
import random
import numpy as np
from ticktock import *

def alias_sampler(_ps):
    tic("setup")
    ps = np.copy(_ps)
    ps /= np.sum(ps)
    n = len(ps)
    alias = np.zeros(n)
    prob = np.zeros(n)
    ps *= n
    toc()
    # could vectorize this?...
    tic("initializing small and large")
    # for i in xrange(n):
    #     if ps[i] < 1:
    #         small.append(i)
    #     else:
    #         large.append(i)
    small = ps[ps<1]
    large = ps[ps>=1]
    toc()
    tic("while small and large")
    sidx = 0
    lidx = 0
    sn = len(small)
    ln = len(large)
    #while np.any(small) and np.any(large):
    while sidx < len(small) and lidx < len(large):
        # l = small.pop()
        # g = large.pop()
        s = small[sidx]
        l = large[lidx]
        pl = ps[l]
        prob[l] = pl
        alias[l] = g
        ps[g] += pl - 1
        if ps[g] < 1:
            #small.append(g)
            np.append(small,g)
        else:
            #large.append(g)
            np.append(large,g)
        sidx += 1
        lidx += 1
    toc()
    tic("while large")
    while large:
        g = large.pop()
        prob[g] = 1
    toc()
    tic("while small")
    while small:
        l = small.pop()
        prob[l] = 1
    toc()
    def sampler():
        i = random.randrange(n)
        if random.random() < prob[i]:
            return i
        else:
            return alias[i]
    return sampler

def compare_alias_and_inverse_cdf():
    from project_utils import inverse_cdf_sampler
    import time
    from utils import qqplot
    from ticktock import tic,toc
    G = 5000000
    q = 50
    num_samples = 1000000
    ps = [random.random() * 50/float(G) for i in range(G)]
    tic("inverse sampler")
    inv_sampler = inverse_cdf_sampler(ps)
    toc()
    tic("inverse sampling")
    inv_samples = [inv_sampler() for i in range(num_samples)]
    toc()
    tic("alias sampler")
    al_sampler = alias_sampler(ps)
    toc()
    tic("alias sampling")
    al_samples = [al_sampler() for i in range(num_samples)]
    toc()
    return inv_samples,al_samples
