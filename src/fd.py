from math import log,exp
from utils import inverse_cdf_sample
import random,itertools
from ssa import inverse_cdf_sample_unnormed
from numpy import random as nprandom
import numpy as np
from sample import direct_sampling_ps,make_sampler,ss_from_xs
from mean_field import dpois_binom
beta = 1

def fd_solve(koffs,mu,beta=1):
    eps = [log(k)/beta for k in koffs] # note we drop the negative because ks are for OFF-rates
    return [1/(1+exp(beta*(ep-mu))) for ep in eps]

fd_func = np.vectorize(lambda ep,mu:(1.0/(1+exp((ep-mu)))))

def fd_solve_np_ref(eps,mu,beta=1):
    #koffs = np.array(koffs)
    return fd_func(eps,mu)

def fd_solve_np(all_eps,mu):
    fwd_eps,rev_eps = all_eps
    eps = np.log(np.exp(fwd_eps) + np.exp(fwd_eps))
    return 1/(1+np.exp(eps-mu))
    
def rfd_ref(ps):
    return [int(random.random() < p) for p in ps]

def rfd_xs(ps):
    xs = []
    for i,p in enumerate(ps):
        if random.random() < p:
            xs.append(i)
    return xs

def rfd_xs_np(ps):
    rs = nprandom.uniform(size=len(ps))
    return np.nonzero(rs < ps)[0]

def rfd_xs_np_spec(ps):
    pass
    
def rbernoulli(p):
    """Sample a bernoulli random variable with mean p using as few bits as
    possible, inverse arithmetic coding"""
    # The idea is to sample a random real r in the unit interval, one
    # bit (i.e. binary decimal place) at a time, until we are sure
    # that either r < p or r > p.
    hi = 1.0
    lo = 0.0
    d = -1
    while lo < p < hi:
        if random.getrandbits(1):
            lo = (hi + lo)/2
        else:
            hi = (hi + lo)/2
        print lo,hi
    if p > hi:
        return 1
    else:
        return 0
    
def log_dfd(config,ps):
    return sum([log(p if c else (1-p)) for c,p in zip(config,ps)])

def roccupancy_ref(ps):
    return sum(random.random() < p for p in ps)

def roccupancy(ps):
    n = 0
    for p in ps:
        if random.random() < p:
            n += 1
    return n
    
def estimate_entropy(ps,n=1000):
    return mean([show(-log_dfd(rfd_ref(ps),ps)) for i in range(n)])

def fd_entropy(ps):
    return -sum(p*log(p) + (1-p)*log(1-p) for p in ps)
    
def rfd_spec(ps,block_length=10000,verbose=False):
    G = len(ps)
    config = []
    total_rands = 0
    for block in xrange(0,G,block_length):
        block_ps = ps[block:block+block_length]
        block_prob = 1 - product([1-p for p in block_ps]) # probability of at least one hit
        if verbose:
            print "block prob:",block_prob
        total_rands += 1
        if random.random() > block_prob:
            config.extend([0]*len(block_ps))
            if verbose:
                print "avoided sampling"
        else:
            block_config = rfd_ref(block_ps)
            total_rands += len(block_ps)
            i = 1
            while sum(block_config) == 0:
                block_config = rfd_ref(block_ps) # must resample until getting at least one hit
                total_rands += len(block_ps)
                i += 1
            if verbose:
                print "had to sample %s times" % i
            config.extend(block_config)
    print "total_rands:",total_rands
    return config

def rfd_spec2(ps):
    config = []
    block_start = 0
    acc = 0
    for block_stop,p in enumerate(ps):
        acc += p
        if acc > 0.5:
            print block_stop
            hit_in_block = random.random() < acc
            if hit_in_block:
                block = rfd_ref(ps[block_start:block_stop])
            else:
                block = [0]*(block_stop - block_start)
            config.append(block)
            acc = 0
    return config

def config_from_int(i,G):
    assert 0 <= i < 2**G
    raw_bin_string = bin(i)[2:]
    bin_string = ('0'*(G-len(raw_bin_string))) + raw_bin_string
    return [int(c) for c in bin_string]
    

def rfd_spec3(ps):
    """sample config by arithmetic coding scheme"""
    G = len(ps)
    lo = 0
    hi = 2**G
    
def rfd_poisson(ps,n):
    """Sample n configs by conditioning on chromosomal occupancy via LeCam's theorem"""
    lam = sum(ps)
    G = len(ps)
    sample_q = lambda:nprandom.poisson(lam) # chromosomal occupancy approximately poisson.
    sampler = make_sampler(ps)
    return [direct_sampling_ps(ps,sample_q(),sampler) for i in xrange(n)]
    
def rfd_pois_binom(ps):
    """Sample a config by conditioning on chromosomal occupancy via
    Poisson binomial distribution (LeCam's theorem approach proved
    unsatisfactory), in blocks"""
    q = inv_cdf_sample_fast(lambda k:dpois_binom(ps,k))
    return direct_sampling_ps(ps,q)

def rfd_pois_binom_rec(ps,block_length=10000):
    block_pss = group_by(ps,block_length)
    pass
    # come back to this
def inv_cdf_sample_fast(f):
    r = random.random()
    k = 0
    p = f(0)
    while r > p:
        k+=1
        p += f(k)
        print k
    return k
