"""
In this file we consider the effects of relaxing the constraint of
copy number.  That is, we consider a Hamiltonian of the form:

E(s) = \sum_i\epsilon_i*s_i + \mu\sum_i s_i
"""
import random
import numpy as np
from math import log,exp,sqrt,pi
import cmath
from project_utils import logfac
from utils import mean,log2,interpolate,pl,product,fac,sd,median,variance
from matplotlib import pyplot as plt

beta = 1

def fd(epsilon,mu,beta=1):
    return 1/(1+exp(beta*(epsilon-mu)))

def probs(eps,mu):
    return [fd(ep,mu) for ep in eps]
    
def log_state_prob(ss,epsilons,mu):
    return sum(map(log,[fd(ep,mu) if s == 1 else 1-fd(ep,mu) for s,ep in zip(ss,epsilons)]))

def chrom_occ(ss):
    return sum(ss)

def mean_occ(eps,mu):
    return sum(fd(ep,mu) for ep in eps)

def var_occ(eps,mu):
    ps = [fd(ep,mu) for ep in eps]
    return sum(p*(1-p) for p in ps)
    
def sd_occ(eps,mu):
    return sqrt(var_occ(eps,mu))

def synth_eps(G):
    return [sum(-2 if random.random() < 0.25 else 0 for j in range(10)) for i in range(G)]
    
def occupancy_distribution(epsilons,mu,verbose=False):
    "Return a matrix v such that v[i] == P(n == i)"
    G = len(epsilons)
    print "computing ps"
    ps = [fd(ep,mu) for ep in epsilons]
    v = [1] + [0]*G
    vnew = v[:]
    J = G + 1
    for i,p in enumerate(ps):
        if verbose:
            if i % 10000:
                print i
        q = 1 - p
        for j in xrange(J):
            #print "j:",j
            if j == 0:
                vnew[j] = p*v[j]
            elif j == G + 1:
                ##print "eliffing"
                vnew[j] = v[j]+q*v[j-1]
            else:
                vnew[j] = p*v[j] + q*v[j-1]
        v = vnew[:]
        #print v,sum(v)
    return list(reversed(v))

def occupancy_distribution_fast(epsilons,mu,verbose=False):
    "Return a matrix v such that v[i] == P(n == i)"
    G = len(epsilons)
    print "computing ps"
    ps = [fd(ep,mu) for ep in epsilons]
    print "finished computing ps"
    v = np.zeros(1+G,dtype=np.float32)
    v[0] = 1
    J = G + 1
    #print "ps:",ps
    for i,p in enumerate(ps):
        #print i
        if verbose:
            if i % 1000 == 0:
                print i
        q = 1 - p
        vq = q*v
        vq[-1] = v[-1]
        #print "vp:",vp
        vp = np.hstack([0,p*v[:-1]])
        #print "vq:",vq
        v = vp + vq
        #print "v:",v
    return v

def occ_dist_gaussian(eps,mu):
    print "computing ps"
    ps = [fd(ep,mu) for ep in eps]
    print "---"
    m = sum(ps)
    s = sqrt(sum(p*(1-p) for p in ps))
    return [dnorm(k,m,s) for k in range(len(ps)+1)]
    
def occ_dist_poisson(eps,mu):
    """Compute probability that occupancy == k, via LeCam's theorem'"""
    print "computing ps"
    return map(exp,log_occ_dist_poisson(eps,mu))

def dpois_binom(ps,k):
    """Return probability that sum of bernoullis given by ps equals k
    according to Poisson binomial distribution.  Complexity is O(len(ps)^2)"""
    n = len(ps)
    C = cmath.exp(2*1j*pi/(n+1))
    return (1.0/(n+1)*sum(C**(-ell*k)*product([1+(C**ell-1)*p for p in ps]) for ell in range(n+1))).real
    
def log_occ_dist_poisson(eps,mu):
    """Compute probability that occupancy == k, via LeCam's theorem'"""
    print "computing ps"
    ps = [fd(ep,mu) for ep in eps]
    print "---"
    lamb = sum(ps)
    return [k*log(lamb) -lamb - logfac(k) for k in range(len(ps)+1)]

def occ_dist_poisson_ref(eps,mus):
    print "computing ps"
    ps = [fd(ep,mu) for ep in eps]
    print "---"
    lamb = sum(ps)
    return [lamb**k*exp(-lamb)/fac(k) for k in range(len(ps)+1)]
    
def rstate(eps,mu):
    return [int(random.random() < fd(ep,mu)) for ep in eps]

def dstate(ss,eps,mu):
    return product([fd(ep,mu) if s else (1-fd(ep,mu)) for s,ep in zip(ss,eps)])

def entropy(eps,mu):
    return sum((lambda p:-(p*log2(p)+(1-p)*log2(1-p)))(fd(ep,mu))
               for ep in eps)

def gaussians(mu_x,sigma_x,n):
    return [random.gauss(mu_x,sigma_x) for i in xrange(n)]
    
def entropy_of_gaussian_ensemble(G,mu_ep,sigma_ep,mu):
    return entropy(gaussians(mu_x,sigma_x,G),mu)

def logit(p):
    return p/(1-p)
    
def dlogitnormal(x,m,s):
    """compute pdf of logit-normal distribution"""
    return 1/(s*sqrt(2*pi))*exp(-(logit(x)-m)**2/(2*s**2))*(1/(x*(1-x)))
    
def main():
    G = 1000
    mu_ep = 0
    sigma_ep = 1
    eps = gaussians(mu_ep,sigma_ep,G)
    mus = interpolate(-100,10,100)
    plt.plot(*pl(lambda mu:mean_occ(eps,mu),mus),label="Mean occ")
    plt.plot(*pl(lambda mu:G/(1+exp(-0.75*mu)),mus),label="predicted occ")
    plt.plot(*pl(lambda mu:sd_occ(eps,mu),mus),label="Sd occ")
    plt.plot(*pl(lambda mu:entropy(eps,mu),mus),label="Entropy (bits)")
    plt.plot([mu_ep,mu_ep],[0,G],linestyle='--')
    plt.plot([mus[0],mus[-1]],[G/2,G/2],linestyle='--')
    plt.xlabel("mu")
    plt.legend()
    plt.show()
    
def projection_experiment():
    """Can we sample by projecting down into subspace where copy number is conserved?"""
    G = 1000
    eps = synth_eps(G)
    mu = -12 # roughly 15-20 copies expected
    sample_n = 100
    samples = []
    ps = [fd(ep,mu) for ep in eps]
    copy_num = 10
    iterations = 0
    successes = 0
    while len(samples) < sample_n:
        iterations += 1
        state = [random.random() < p for p in ps]
        if sum(state) <= copy_num:
            samples.append(state)
            successes += 1
            print successes/float(iterations)
    return samples

def mu_summary_stat_experiment():
    """Can we correlate copy number with a summary statistic?"""
    trials = 100
    ep_mu = -2
    ep_sigma = 5
    G = 100
    ts = []
    copies = []
    eps = [random.gauss(ep_mu,ep_sigma) for i in range(G)]
    mus = interpolate(-10,10,1000)
    eta = mean(eps)
    gamma = 1.0/variance(eps)
    print gamma
    plt.plot(*pl(lambda mu:mean_occ(eps,mu),mus))
    plt.plot(*pl(lambda mu:G*fd(eta,mu,beta=gamma),mus))
    plt.plot(*pl(lambda x:G/2.0,mus))
        
    
        
