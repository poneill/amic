"""
Code for sampling configurations, given a complete binding energy model.
"""

from fd import fd_solve_ss
import numpy as np
from numpy import random as nprandom
from math import exp,log,sqrt
import random
from utils import mh,verbose_gen,fac
from project_utils import inverse_cdf_sampler,log_fac,unsum
import bisect
import scipy

# A binding energy model is a function that assigns a free energy to
# configurations.  The induced probability distribution over
# configurations is then the usual Boltzmann distribution: P(xs) =
# exp(-E(xs))/Z.

# We assume that a configuration vector xs records (1) for the left
# endpoint of each TF and (0) otherwise, and represents a circularized
# chromosome.  We will also assume that any array of genomic scale
# will be represented by a numpy ndarray.

# should theta = (eps,mu,interations) be a tuple?
# should xs be G-len arrays or lists of occupied indices??

def energy_function_from_params(eps,mu,interactions):
    """Given the sequence specific binding potential, chemical potential
    and interaction energies, return a function E(xs) -> (free energy)
    which assigns a free energy to the configuration.  Interactions is
    a list of arbitrary functions of the configuration.
    """
    def energy(xs):
        sequence_specific_energy = np.sum(eps*xs)
        chemical_energy = -np.sum(xs)*mu
        interaction_energy = sum(interaction(xs)
                                 for interaction in interactions)
        return sequence_specific_energy + chemical_energy + interaction_energy
    return energy

def nearest_neighbor_interaction(ep,w):
    """Return an interaction function that assigns an energy of epsilon
    for each pair of tfs lying at a distance of w bases apart."""
    def nn(xs):
        return ep*np.sum(xs*np.roll(xs,w))
    return nn

def count_nns(xs,w):
    """count number of copies lying at a distance of w from each other"""
    nz_entries = xs.nonzero()[0] #take first elem of tuple
    offsets = (nz_entries - np.roll(nz_entries,1)) % G
    return np.count_nonzero(offsets == w)
    
def lift_from_fd(eps,mu,w,epi,interactions,iterations=50000,verbose=0):
    """Given BEM params, sample via lifting from FD."""
    # we include some extraneous parameters in order to harmonize with lift_from_rsa
    G = len(eps)
    E = energy_function_from_params(eps,mu,interactions)
    qs = fd_solve_ss(eps,mu)
    log_1mqs = np.log(1-qs)
    log_qs = np.log(qs) #pull these out on the advice of profiler
    def proposal():
        xs_prime = (nprandom.uniform(size=G) < qs).astype(int)
        log_p = np.sum(log_qs * xs_prime + log_1mqs * (1-xs_prime))
        return xs_prime,log_p
    return mh_with_prob(f=lambda xs:-E(xs),
                        proposal=proposal,
                        iterations=iterations,
                        verbose=verbose)

def gibbs(eps,mu,interactions,iterations):
    G = len(eps)
    xs = np.zeros(G)
    E = energy_function_from_params(eps,mu,interactions)
    for it in xrange(iterations):
        for i in verbose_gen(xrange(G)):
            xs[i] = 1
            on_energy = E(xs)
            xs[i] = 0
            off_energy = E(xs)
            if random.random() < exp(-on_energy)/(exp(-on_energy) + exp(-off_energy)):
                xs[i] = 1
    return xs

def gibbs_fast(eps,mu,w,epi,iterations):
    """Do gibbs sampling assuming an interaction energy of epi at w bases,
    overlap exclusion within w bases"""
    G = len(eps)
    xs = np.zeros(G)
    overlap_penalty = 1000000
    last_placement = None
    for iteration in xrange(iterations):
        for i in (xrange(G)):
            on_energy = eps[i] - mu + epi*xs[(i-w)%G] + overlap_penalty*(i - w > last_placement)
            weight = exp(-on_energy)
            if random.random() < weight/(1+weight):
                xs[i] = 1
                last_placement = i
    return xs

def inverse_cdf_sample(ws):
    r = random.random()
    r *= np.sum(ws)
    acc = 0
    for i,w in enumerate(ws):
        acc += w
        if acc > r:
            return i

def rsa(eps,mu,w,epi,n):
    """do random sequential adsorption of n copies with interaction energy, overlap exclusion"""
    G = len(eps)    
    xs = np.zeros(G)
    ws = np.exp(-eps)
    for copy in xrange(n):
        i = inverse_cdf_sampler(ws)()
        xs[i] = 1
        ws[(i-w)%G] *= exp(-epi)
        ws[(i+w)%G] *= exp(-epi)
        for j in range(-w+1,w):
            ws[(i+j)%G] = 0
    return xs
        
def rsa_with_prob(eps,mu,w,epi,n):
    """Do one round of RSA with n copies, returning state and associated probability mass"""
    G = len(eps)    
    xs = np.zeros(G)
    ws = np.exp(-eps)
    log_prob = 1
    for copy in xrange(n):
        i = inverse_cdf_sampler(ws)()
        #i = inverse_cdf_sampler_fast(ws) # this is only an improvement for large genome sizes...
        log_p = log(ws[i]/np.sum(ws))
        log_prob += log_p
        xs[i] = 1
        ws[(i-w)%G] *= exp(-epi)
        ws[(i+w)%G] *= exp(-epi)
        for j in range(-w+1,w):
            ws[(i+j)%G] = 0
    return xs,log_prob

def inverse_cdf_sampler_fast(ws):
    """sample i from ws, not necessarily normalized"""
    # find greatest i such that cumsum(ws[i:]) < r*W
    n = len(ws)
    W = np.sum(ws)
    lo = 0
    hi = n
    r = random.random() * W
    lo_sum = 0
    hi_sum = W
    while lo + 1!= hi:
        guess = int((lo+hi)/2)
        guess_sum = lo_sum + np.sum(ws[lo:guess])
        if guess_sum < r:
            lo = guess
            lo_sum = guess_sum
        else:
            hi = guess
            hi_sum = guess_sum
    return lo

def rsa_with_prob2(eps,mu,w,epi,n):
    """Do one round of RSA with n copies, returning state and associated probability mass"""
    G = len(eps)    
    xs = np.zeros(G)
    ws = np.exp(-eps)
    #ws_copy = np.copy(ws)
    cum_ws = np.cumsum(ws)
    log_prob = 1
    for copy in xrange(n):
        i = cum_sampler(cum_ws)()
        log_p = log(ws[i]/cum_ws[-1])
        log_prob += log_p
        xs[i] = 1
        #ws_copy[(i-w)%G] *= exp(-epi)
        cum_ws[(i-w)%G:] += ws[i-w]*(exp(-epi)-1)
        #print "1",np.linalg.norm(#ws_copy - unsum(cum_ws))
        #ws_copy[(i+w)%G] *= exp(-epi)
        cum_ws[(i+w)%G:] += ws[i+w]*(exp(-epi)-1)
        #print "2",np.linalg.norm(#ws_copy - unsum(cum_ws))
        # for j in range(-w+1,w):
        #     cum_ws[(i+j)%G] = 0
        u,v = (i+(-w+1))%G, (i+(w))%G
        a,b = cum_ws[u-1],cum_ws[v-1]
        cum_ws[u:v] = a
        cum_ws[v:] -= (b-a)
        ### sanity checking
        #ws_copy[(i-w)%G] *= exp(-epi)
        #ws_copy[(i+w)%G] *= exp(-epi)
        # for j in range(-w+1,w):
        #     pass
            #ws_copy[(i+j)%G] = 0
        #print "3",np.linalg.norm(#ws_copy - unsum(cum_ws))
        ## end sanity check
    return xs,log_prob

def cum_sampler(cum_ws):
    """make a bintree for sampling from discrete distribution ps over set xs"""
    total = cum_ws[-1]
    def sampler():
        r = random.random() * total
        i = bisect.bisect_left(cum_ws,r)
        return i
    return sampler

def lift_from_rsa(eps,mu,w,epi,interactions,rsawp=rsa_with_prob,iterations=50000,verbose=0):
    """Given BEM params, sample via lifting from RSA."""
    G = len(eps)
    E = energy_function_from_params(eps,mu,interactions)
    def proposal():
        """First sample copy number, then RSA a config with that copy number"""
        ps = fd_solve_ss(eps,mu)
        copy_num,log_copy_mass = propose_copy_number_poisson(ps)
        config,log_rsa_mass = rsawp(eps,mu,w,epi,copy_num)
        return config,log_rsa_mass + log_copy_mass
    print "entering chain"
    return mh_with_prob(f=lambda xs:-E(xs),
                        proposal=proposal,
                        iterations=iterations,
                        verbose=verbose)

def propose_copy_number_gauss(ps):
    """From a vector of fd ps, sample a copy number using gaussian approximation"""
    mean = np.sum(ps)
    var = np.sum(ps*(1-ps))
    print "mean: %s, var:%s",(mean,var)
    return int(random.gauss(mean,sqrt(var)))

def propose_copy_number_poisson(ps):
    """From a vector of fd ps, sample a copy number using poisson
    approximation.  Idea: if occupation is rare, i.e. most ps are <<1,
    then sum(ps*(1-ps)) ~= (sum_ps).  
    """
    m = np.sum(ps)
    lamb = 1.0/m
    k = scipy.stats.poisson.rvs(m)
    log_mass = k*log(lamb) - log_fac(k) - lamb # poisson probability mass
    return k,log_mass
    
def mh_with_prob(f,proposal,iterations=50000,modulus=1000,verbose=False,every=1):
    """Metropolis-Hastings sampler, assuming that proposal function
    returns density of proposed, state, e.g.  xp, p = proposal().
    Also assume that proposal is independent of current state, and
    use_log=True.  Rather than returning a list of configurations
    (which is memory intensive), return a mean configuration.

    """
    x,log_prob = proposal()
    xs = x
    fx = f(x)
    acceptances = 0
    proposed_improvements = 0
    try:
        for it in xrange(iterations):
            if it % modulus == 0:
                print it,fx,"acceptance ratio:",acceptances/float(max(it,1)),"occupancy:",np.sum(x)
            x_new, log_prob_new = proposal()
            fx_new = f(x_new)
            ratio = (fx_new - fx) + (log_prob - log_prob_new)
            r = log(random.random())
            if verbose and it % verbose == 0:
                ratio_string = "ratio: %s - %s + %s - %s = %s" % (fx,fx_new,log_prob,log_prob_new,ratio)
                print it,"fx:",fx,"fx_new:",fx_new,ratio_string,"r:",r,\
                       "accept" if ratio > r else "no accept","acceptance ratio:",acceptances/float(max(it,1))
            if r < ratio:
                x = x_new
                fx = fx_new
                log_prob = log_prob_new
                acceptances += 1
            if it % every == 0:
                xs += x
        if verbose > 0:
            print "Proposed improvement ratio:",proposed_improvements/float(iterations)
        print "Acceptance Ratio:",acceptances/float(iterations)
        return xs/float(iterations+1)
    except KeyboardInterrupt:
        if verbose > 0:
            print "Proposed improvement ratio:",proposed_improvements/float(iterations)
        print "Acceptance Ratio:",acceptances/float(it)
        return xs/float(it+1)
        
