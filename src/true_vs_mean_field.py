from math import *
beta = 1
from transfer_matrix_reduced import occupancies_ref
from mean_field import probs,mean_occ,rstate
from scipy import polyfit
from utils import argmax,argmin
from project_utils import falling_fac

def eps_from_ks(ks):
    return [-log(k)/beta for k in ks]

def ks_from_eps(eps):
    return [exp(-beta*ep) for ep in eps]

def find_mean_field_approximation(ks,q,ps=None):
    eps = eps_from_ks(ks)
    if ps is None:
        ps = occupancies_ref(ks,q)
    mu_min = -10
    mu_max = 10
    mu_steps = 1000
    mus = interpolate(mu_min,mu_max,mu_steps)
    coeffs = map(lambda mu:polyfit(ps,probs(eps,mu),1)[0],
                 mus)
    max_coeff_idx = argmax(coeffs)
    # find mu corresponding to best fit.  Cutoff at peak, since curve
    # has two intersections with y = 1.
    mu_idx = argmin(map(lambda coeff:(1-coeff)**2,coeffs[:max_coeff_idx]))
    best_mu = mus[mu_idx]
    qs = probs(eps,best_mu)
    print "best_mu:",best_mu
    print "mean copy number: %s (sd: %s) vs. sum(ps) %s" %(mean_occ(eps,best_mu),sd_occ(eps,best_mu),sum(ps))
    print "pearson correlation:",pearsonr(ps,qs)
    print "best linear fit: p = %s*q + %s" % tuple(polyfit(qs,ps,1))
    return best_mu

def free_energy(eps,q,mu,samples=1000,use_annealed_approx=False):
    """Compute free energy for true distribution given best approximation.  Return beta*free_energy,actually"""
    # def Q(xs):
    #     return product(fd(ep,mu) if x else 1 - fd(ep,mu) for x,ep in zip(xs,eps))
    Sq = -sum((p*log(p)+(1-p)*log(1-p) for p in probs(eps,mu)))
    def E(xs):
        """Compute energy function for P,unnormalized"""
        ff = falling_fac(q,sum(xs))
        if ff == 0:
            return 1000
        else:
            return log(ff) + sum(ep*x for (x,ep) in zip(xs,eps))
    if use_annealed_approx: # to get around impossible states where E=\infty
        mean_E = log(mean(exp(E(rstate(eps,mu))) for i in range(samples))) # take <E(xs)>_Q
    else:
        mean_E = mean(E(rstate(eps,mu)) for i in range(samples)) # take <E(xs)>_Q
    return beta*mean_E - Sq
              
def synth_eps(mu0,sigma0,mu1,sigma1,n,G):
    """Return epsilons from two component mixturd distribution containing n sites, G-n background sites"""
    eps0 = [random.gauss(mu0,sigma0) for i in xrange(n)]
    eps1 = [random.gauss(mu1,sigma1) for i in xrange(G-n)]
    return eps0 + eps1
