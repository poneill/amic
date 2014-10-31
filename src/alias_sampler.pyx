"""
Implement Vose's method for alias sampling

"""
import numpy as np

def alias_sampler(_ps):
    ps = np.copy(_ps)
    ps /= np.sum(ps)
    n = len(ps)
    alias = np.zeros(n)
    prob = np.zeros(n)
    ps *= n
    small = []
    large = []
    # could vectorize this?...
    for i in range(n):
        if ps[i] < 1:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        l = small.pop()
        g = large.pop()
        pl = ps[l]
        pg = ps[g]
        prob[l] = pl
        alias[l] = g
        ps[g] += pl - 1
        if ps[g] < 1:
            small.append(g)
        else:
            large.append(g)
    while large:
        g = large.pop()
        prob[g] = 1
    while small:
        l = small.pop()
        prob[l] = 1
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
    G = 5000000
    q = 50
    num_samples = 1000000
    ps = [random.random() * 50/float(G) for i in range(G)]
    tic = time.time()
    inv_sampler = inverse_cdf_sampler(ps)
    toc = time.time()
    print "inverse transform sampler in %s seconds" % (toc - tic)
    tic = time.time()
    inv_samples = [inv_sampler() for i in range(num_samples)]
    toc = time.time()
    print "inverse transform sampling in %s seconds" % (toc - tic)
    tic = time.time()
    al_sampler = alias_sampler(ps)
    toc = time.time()
    print "alias sampler in %s seconds" % (toc - tic)
    tic = time.time()
    al_samples = [al_sampler() for i in range(num_samples)]
    toc = time.time()
    print "alias sampling in %s seconds" % (toc - tic)
