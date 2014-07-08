from utils import inverse_cdf_sample,mean,transpose,mh,bisect_interval,verbose_gen,concat,fac,maybesave
from mean_field import rstate,dstate,fd
from project_utils import inverse_cdf_sampler
from collections import Counter
from utils import product
from project_utils import powsum,esp
import time

def sequential_sample_ref(ks,q):
    G = len(ks)
    chromosome = [0]*(G+1)
    new_ks = ks[:] + [1]
    for tf in range(q):
        #print "new_ks:",new_ks,"q:",q
        Z = float(sum(new_ks))
        pos = inverse_cdf_sample(range(G+1),[k/Z for k in new_ks])
        #print "pos:",pos
        chromosome[pos] += 1
        if chromosome[pos] > 1 and pos < G:
            raise Exception("Chromosome[%s] > 1" % pos)
        if pos < G:
            new_ks[pos] = 0
    return chromosome

def gibbs_sample(ks,xs):
    """
    ks: a G-length vector of rate constants
    xs: a q-length vector such that x_i contains the position of the ith TF ([0 to G-1]), or G if off-chromosome"""
    xs_new = xs[:]
    #print "xs_new:",xs_new
    G = len(ks)
    #print "G:",G
    q = len(xs)
    #print "q:",q
    for j in range(q):
        cur_pos = xs_new[j]
        cur_ks = [(k if (i not in xs_new or i == cur_pos) else 0) for i,k in enumerate(ks)] + [1]
        cur_Z = float(sum(cur_ks))
        sampler = inverse_cdf_sampler(range(G+1),[cur_k/cur_Z for cur_k in cur_ks])
        new_pos = sampler()
        xs_new[j] = new_pos
    return xs_new

def metropolis(ks,q,verbose=False,mu_offset=0,iterations=50000):
    """Metropolis-Hastings sampling for ks, given product-bernoulli proposal function"""
    G = len(ks)
    eps = [-log(k) for k in ks]
    f = lambda mu:sum(fd(ep,mu) for ep in eps) - q
    mu = bisect_interval(f,-50,50) + mu_offset
    def weight(ss):
        return (falling_fac(q,sum(ss))*product(k**s for k,s in zip(ks,ss)))
    def proposal(ss):
        #state = [int(random.random() < p) for _ in range(len(ss))]
        state = rstate(eps,mu)
        #print "proposed state with occ:",sum(state)
        return state
    def dprop(ss):
        prop = dstate(ss,eps,mu)
        #print "prop:",prop 
        return prop        
    x0 = proposal([0] * len(ks))
    return mh(weight,proposal,x0,dprop=dprop,verbose=verbose,iterations=iterations)
    
def gibbs_sample_fast(ks,xs,iterations,cur_ks=None,cur_Z=None):
    """
    ks: a G-length vector of rate constants
    xs: a q-length vector such that x_i contains the position of the ith TF ([0 to G-1]), or G if off-chromosome"""
    xs_new = xs[:]
    cur_ks = ks[:]
    cur_Z = float(sum(cur_ks))
    G = len(ks)
    q = len(xs)
    for x in xs_new:
        if x < G:
            cur_ks[x] = 0
            cur_Z -= ks[x]
    for iteration in xrange(iterations):
        #print iteration
        for j in range(q):
            cur_pos = xs_new[j]
            if cur_pos < G:
                cur_ks[cur_pos] = ks[cur_pos]
                cur_Z += ks[cur_pos]
            sampler = inverse_cdf_sampler(range(G+1),[cur_k/cur_Z for cur_k in cur_ks])
            new_pos = sampler()
            xs_new[j] = new_pos
            if new_pos < G:
                cur_ks[new_pos] = 0
                cur_Z -= ks[new_pos]
    return xs_new

def gibbs_fast_harness(ks,q,iterations):
    G = len(ks)
    xs = [G]*q
    return gibbs_sample_fast(ks,xs,iterations=iterations)
    
def gibbs_sample_iterate(ks,xs,iterations):
        xs_new = xs[:]
        for iteration in range(iterations):
            xs_new = gibbs_sample(ks,xs_new)
        return xs_new
        
def ss_from_xs(xs,G):
    """Project a TF-coordinate vector down into configuration space."""
    return [xs.count(i) for i in range(G+1)]

def sequential_sample_many(ks,q,n):
    return map(mean,transpose([sequential_sample_ref(ks,q) for i in verbose_gen(xrange(n))]))

def gibbs_sample_many(ks,q,t,n):
    """Sample system (ks,q) by gibbs sampling at time t, for n trials"""
    G = len(ks)
    return map(mean,transpose([ss_from_xs(gibbs_sample_iterate(ks,[G]*q,t),G)
                               for i in verbose_gen(xrange(n))]))

def physics_experiment():
    """Track order parameters of gibbs sampling"""
    G = 100
    mean_energy = 0
    sigma_energy = 1
    epsilons = [random.gauss(mean_energy,sigma_energy) for i in range(G)]
    beta = 1 # inverse temp set to unity
    ks = [exp(-beta*ep) for ep in epsilons]
    q = 10
    iterations = 10000
    xs = [G]*q # initial state has all copies off-chromosome
    def x_energy(xs):
        return sum(epsilons[x] if x < G else 0 for x in xs)
    energies = [x_energy(xs)]
    for iteration in xrange(iterations):
        xs = gibbs_sample(ks,xs)
        energies.append(x_energy(xs))
    return energies

def how_bad_is_sequential_sampling():
    G = 100
    ks = [1]*G
    q = 10
    
def reservoir_sample(ks,q):
    reservoir = [] # reservoir is a bag of positions bound by tfs
    cur_Z = 1.0
    free = q
    for i,k in enumerate(ks):
        cur_Z += free*k
        for tf in range(free):
            r = random.random()
            if r < k/cur_Z: # accept current item
                if free == 0:
                    reservoir.remove(random.choice(reservoir))
                reservoir.append(i)
                free -= 1
                break
    return reservoir

def direct_sampling_ref(ks,q):
    """ks is a vector of the form [k0,k1,kg], i.e. k0 = 1"""
    Z = float(sum(ks))
    G = len(ks)
    ps = [k/Z for k in ks]
    sampler = inverse_cdf_sampler(range(len(ks)),ps)
    while True:
        ss = [sampler() for j in range(q)]
        counts = Counter(ss)
        if all(counts[i] <= 1 for i in range(1,G+1) if i > 0):
            return ss

def direct_sampling(ks,q,sampler=None,verbose=False,debug_efficiency=False):
    """ks is a vector of the form [k0,k1,kg], i.e. off-rate k0 = 1"""
    if sampler is None:
        sampler = make_sampler(ks)
    iterations = 0
    while True:
        if verbose and iterations % 1000 == 0:
            print iterations
        iterations +=1
        ss = [sampler() for j in range(q)]
        #print "ss:",ss
        counts = Counter(ss)
        if all(counts[i] <= 1 for i in counts if i > 0):
            efficiency = 1.0/iterations
            if verbose:
                print "direct sampling efficiency:",efficiency
            if debug_efficiency:
                return efficiency
            else:
                return ss

def make_sampler(ks):
    """Return an efficient bisection sampler for inverse cdf sampling
    from ks.
    Usage:
    >>> sampler = make_sampler(ks)
    >>> x = sampler()
    """
    Z = float(sum(ks))
    ps = [k/Z for k in ks]
    return inverse_cdf_sampler(range(len(ks)),ps)

def rsa(ks,q,sampler=None,verbose=False,debug_efficiency=False):
    """Perform random sequential adsorption.  Note: Not a method for
    sampling from equilibrium distribution!  ks is a vector of the
    form [k0,k1,kg], i.e. off-rate k0 = 1.
    
    """
    if sampler is None:
        sampler = make_sampler(ks)
    ss = []
    for j in range(q):
        accepted_yet = False
        while not accepted_yet:
            s = sampler()
            if s == 0 or not s in ss:
                ss.append(s)
                accepted_yet = True
    return ss

def smart_rsa(ks,q,sampler=None,verbose=False,debug_efficiency=False):
    """Perform random sequential adsorption without rejection.  Note:
    Not a method for sampling from equilibrium distribution!  ks is a
    vector of the form [k0,k1,kg], i.e. off-rate k0 = 1.
    
    """
    ss = []
    N = len(ks)
    _ks = ks[:]
    for j in range(q):
        s = inverse_cdf_sample(range(N),normalize(_ks))
        ss.append(s)
        if s > 0:
            _ks[s] = 0
    return ss

def occs_from_direct_sampling(samples,ks):
    """Given a _list_ of samples and ks, compute occupancies"""
    num_samples = float(len(samples))
    counts = Counter(concat(samples))
    G = len(ks)
    return [counts[i]/num_samples for i in range(G)]
            

ncp_dict = {}
def ncp(ps,n):
    """Compute non-collision probability for n draws from distribution
    ps according to """
    def _ncp(ps,n):
        """find non-coincidence probability for n draws from ps"""
        tic = time.time()
        #print n,ncp_dict
        if n == 0:
            return 1
        if n not in ncp_dict:
            #print "didn't find",n
            ans = sum((-1)**(i-1)*fac(n-1)/fac(n-i)
                      *powsum(ps,i)*_ncp(ps,n-i) for i in range(1,n+1))
            ncp_dict[n] = ans
        else:
            #print "found",n
            pass
        toc = time.time()
        #print "leaving call w/ %s in %1.2f sec" % (n,toc-tic)
        return ncp_dict[n]
    return _ncp(ps,n)

def ncp2(ps,n):
    return fac(n)*esp(ps,n)/1.0

def est_ncp(ps,n,trials,verbose=False):
    accs = 0
    A = range(len(ps))
    for trial in xrange(trials):
        if verbose:
            if trial % verbose == 0:
                print "%s/%s\r" % (trial,trials),
                sys.stdout.flush()
        xs = [inverse_cdf_sample(A,ps) for i in range(n)]
        if len(set(xs)) == n:
            accs += 1
    return accs/float(trials)

def graph_acceptance_ratio(filename):
    ps = normalize([exp(-random.gauss(0,5)) for i in xrange(5000000)])
    ars = [show(ncp2(ps,i)) for i in range(10+1)]
    plt.plot(ars)
    plt.semilogy()
    plt.xlabel("Copy Number")
    plt.ylabel("Acceptance Ratio")
    plt.title("Acceptance Ratio vs. Copy number for 5*10^6 LN(0,5) sites")
    maybesave(filename)
    
def check_ncp(ps,ns,trials):
    ncps = [ncp(ps,n) for n in ns]
    est_ncps = [est_ncp(ps,n,trials) for n in ns]
    errs = [1.96*sqrt(p*(1-p)/float(trials)) for p in est_ncps]
    plt.plot(ns,ncps)
    plt.errorbar(ns,est_ncps,yerr=errs)
    plt.show()
    
