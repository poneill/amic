import random
import bisect
import itertools
from utils import rslice,product,verbose_gen,fac,log,mean,transpose,zipWith,h,simplex_sample,choose,fac
from math import exp,factorial,sqrt,pi
import numpy as np
import cmath
import numpy as np
from cumsum import cumsum
from scipy.special import gammaln
from time import ctime

def random_site(n):
    """Generate a random sequence of n nucleotides"""
    return "".join(random.choice("ACGT") for i in range(n))

def inverse_cdf_sample(xs,ps):
    """Sample from discrete distribution ps over set xs"""
    r = random.random()
    acc = 0
    for x,p in zip(xs,ps):
        acc += p
        if acc > r:
            return x

def cumsum_ref(xs):
    """Replaced by cython: cumsum.pyx"""
    acc = 0
    acc_list = [0]*len(xs)
    for i,x in enumerate(xs):
        acc += x
        acc_list[i] = acc
    return acc_list
    
def inverse_cdf_sampler(ps):
    """make a bintree for sampling from discrete distribution ps over set xs"""
    #cum_ps = cumsum(ps)
    cum_ps = np.cumsum(ps)
    total = cum_ps[-1]
    def sampler():
        r = random.random() * total
        i = bisect.bisect_left(cum_ps,r)
        return i
    return sampler

def rejection_sampler(ps):
    N = len(ps)
    def sampler():
        while True:
            i = random.randrange(N)
            if random.random() < ps[i]:
                return i
    return sampler

def alias_sampler(ps):
    n = len(ps)
    alias = [0]*n
    prob = [0]*n
    T = [n*p for p in ps]
    for j in range(1,n):
        pl = indices.pop()
        g = indices.pop(-1)
        prob[l] = n*ps[l]
        alias[l] = g
        
    
def test_inverse_cdf_sampler():
    K = int(5*10**6)
    trials = K
    ps = [1.0/K for i in xrange(K)]
    sampler = inverse_cdf_sampler(ps)
    samples = [sampler() for i in verbose_gen(xrange(trials),modulus=100000)]
    plt.hist(samples,bins=1000)
    plt.show()
    
def pprint(xs):
    """Pretty print a nested list"""
    for x in xs:
        print x
    
def rpois(_lambda):
    """Sample Poisson rv"""
    L = exp(-_lambda)
    k = 0
    p = 1
    while p > L:
        k += 1
        u = random.random()
        p *= u
    return k - 1

def dpois(k,_lambda):
    """pmf of Poisson rv"""
    return exp(-_lambda)*_lambda**k/factorial(k)

def mean(xs):
    return sum(xs)/float(len(xs))

def variance(xs,correct=True):
    n = len(xs)
    correction = n/float(n-1) if correct else 1
    mu = mean(xs)
    return correction * mean([(x-mu)**2 for x in xs])

def sd(xs,correct=True):
    return sqrt(variance(xs,correct=correct))

def pearsonr(xs,ys):
    mu_x = mean(xs)
    mu_y = mean(ys)
    return (mean([(x-mu_x)*(y-mu_y) for (x,y) in zip(xs,ys)])/
            (sd(xs,correct=False)*sd(ys,correct=False)))

def normalize(xs):
    total = float(sum(xs))
    return [x/total for x in xs]

def scores_to_probs(scores):
    """Compute probabilities from energy scores using Boltzmann statistics"""
    exp_scores = [exp(-score) for score in scores]
    Z = sum(exp_scores)
    return [exp_score/Z for exp_score in exp_scores]

def falling_fac(n,k):
    return product([n-j for j in range(k)])
    
def esp_ref(ks,j):
    """compute jth elementary symmetric polynomial on ks"""
    n = len(ks)
    return sum(product(rslice(ks,comb)) for comb in itertools.combinations(range(n),j))

def powsum(ps,k):
    return sum(p**k for p in ps)

def powsums_ref(ps,k):
    return [powsum(ps,i) for i in range(k+1)]

def powsums(ps,k):
    qs = [1]*len(ps)
    psums = [sum(qs)]
    for i in xrange(k):
        qs = [q*p for (q,p) in zip(qs,ps)]
        psums.append(sum(qs))
    return psums
    
def powsums_np(ps,k):
    ps_arr = np.array(ps)
    qs = np.ones(len(ps))
    psums = [sum(qs)]
    for i in xrange(k):
        qs = qs * ps_arr
        psums.append(np.sum(qs))
    return psums
    
def esp_ref2(ps,k):
    if k == 0:
        return 1
    else:
        return sum((-1)**(i-1) *esp(ps,k-i)*powsum(ps,i)
                   for i in range(1,k+1))/float(k)

def esp(ps,k,powsums=None):
    print "calling esp(ps,%s)" % k
    if k == 0:
        return 1
    if powsums is None:
        powsums = [powsum(ps,i) for i in range(k+1)]
        #print powsums
    return sum((-1)**(i-1)*esp(ps,k-i,powsums)*powsums[i]
                   for i in range(1,k+1))/float(k)


def esp_spec(ps,k,powsums=None):
    #print "calling esp(ps,%s)" % k
    if k == 0:
       return 1
    if powsums is None:
        print "computing powersums..."
        powsums = [powsum(ps,i) for i in verbose_gen(range(k+1))]
        print "finished with powersums"
    esp_array = [None]*(k+1)
    esp_array[0] = 1
    for cur_k in range(1,k+1):
        ans = sum((-1)**(i-1)*esp_array[cur_k-i]*powsums[i]
                  for i in range(1,cur_k+1))/float(cur_k)
        esp_array[cur_k] = ans
        #print esp_array
    return esp_array[k]
    
def logfac(n):
    return sum(log(i) for i in range(1,n+1))

def score_seq(matrix,seq,ns=False):
    beta = 1
    """Score a sequence with a motif."""
    base_dict = {'A':0,'C':1,'G':2,'T':3}
    ns_binding_const = -8 #kbt
    #specific_binding = sum([row[base_dict[b]] for row,b in zip(matrix,seq)])
    specific_binding = 0
    for i in xrange(len(matrix)):        
        specific_binding += matrix[i][base_dict[seq[i]]]
    if ns:
        return log(exp(-beta*specific_binding) + exp(-beta*ns_binding_const))/-beta
    else:
        return specific_binding

def score_genome(matrix,seq):
    w = len(matrix)
    G = len(seq)
    return [score_seq(matrix,seq[i:i+w]) for i in range(G-w+1)]

def sample_average_ref(sample):
    return map(mean,transpose(sample))

def sample_average(sample):
    if type(sample[0]) is np.ndarray:
        return mean(sample)
    G = len(sample[0])
    n = float(len(sample))
    avg = [0]*G
    for ss in sample:
        for i,s in enumerate(ss):
            if s:
                avg[i] += 1.0/n
    return avg

def sample_average_from_xss(xss,G):
    """Return sample average from x-configs"""
    avg = [0]*G
    n = float(len(xss))
    for xs in xss:
        for x in xs:
            avg[x] += 1.0/n
    return avg
    
def _dbg(obj,debug):
    if debug:
        print obj
        
def wrs(ws,k,debug=False,return_rs=False):
    """Weighted reservoir sampling for k items"""
    _dbg = lambda x:_dbg(x,debug)
    js = []
    rs = []
    for j,w in enumerate(ws):
        r = random.random()**(1.0/w)
        #dbg( "considering j:%s, w:%s, r:%s" % (j,w,r))
        if j < k:
            #dbg( "preinserting j:%s, w:%s, r:%s" % (j,w,r))
            pos = bisect.bisect(rs,r)
            js.insert(pos,j)
            rs.insert(pos,r)
        else:
            #dbg( "comparing r: %s to rs:%s" % (r,rs))
            pos = bisect.bisect(rs,r)
            #dbg( "pos: %s" % pos)
            if pos > 0:
                #dbg( "inserting")
                js.insert(pos,j)
                rs.insert(pos,r)
                js = js[-k:]
                rs = rs[-k:]
    if return_rs:
        return js,rs
    else:
        return js
    
    
def wrs2(ws,k):
    js,rs = wrs(ws,k,return_rs=True)
    print js,rs
    return [j for j,r in zip(js,rs) if random.random() < r]

def compute_partition(ks,q):
    """Compute partition function for G-q model by direct enumeration"""
    return sum(falling_fac(q,j)*esp(ks,j) for j in range(q+1))

def compute_coefficients_ref(ks):
    """given ks, the roots of a polynomial P(x) = a_n x^n+...+a_1x^1+a_0,
    compute sequence of coefficients a_n...a_0"""
    coeffs = [1]
    for k in ks:
        coeffs = zipWith(lambda x,y:x+y,coeffs+[0],[0]+[-k*c for c in coeffs])
    return coeffs

def compute_coefficients(ks):
    arr = np.array([0 for i in ks] + [0])
    arr[-1] = 1
    for k in verbose_gen(ks,modulus=10000):
        #print np.roll(arr,-1), - k*arr
        arr = np.roll(arr,-1) - k*arr
        #print arr
    return arr

def data_bunch(xs):
    """Alg 1.28 of Krauth"""
    L = len(xs)
    assert log2(L) == int(log2(L)) # len must be power of 2
    ys = xs[:]
    ses = []
    while len(ys) > 1:
        ses.append(se(ys))
        ys = map(mean,group_by(ys,2))
    return ses

def thin(xs,k):
    return xs[::k]

def udist(n):
    """Return uniform distribution on n outcomes"""
    return normalize([1]*n)

def hdist(desired_ent,n):
    """Return distribution on n outcomes with entropy <= h (bits)"""
    ent = n
    while(ent > desired_ent):
        ps = simplex_sample(n)
        ent = h(ps)
    return ps
        
def leps_from_config(config):
    """From a binary vector of length G representing a configuration, return left endpoints of tfs"""
    return [i for i,c in enumerate(config) if c]

def np_normalize(arr):
    return arr/np.sum(arr)

def dbinom(k,N,p):
    """Compute probability of k out N successes at probability p"""
    return choose(N,k)*p**k*(1-p)**(N-k)

def log_dbinom(k,N,p):
    """Compute log probability of k out N successes at probability p"""
    return log(choose(N,k)) + k*log(p) + (N-k)*log(1-p)

def log_choose(N,k):
    return log(choose(N,k))

def log_choose_approx(N,k):
    return k*log(N) - log(fac(k))
    
def log_dbinom_approx(k,N,p):
    """Compute log probability of k out N successes at probability p"""
    return log_choose_approx(N,k) + k*log(p) + (N-k)*log(1-p)

half_log2_pi = 0.5*log(2*pi)

def log_fac(n):
    if n <= 1:
        return 0
    else:
        return (n+0.5)*log(n) - n + half_log2_pi #n*log(n) - n + 0.5*log(2*pi*n)
    # if n < 171:
    #     return log(fac(n))
    # else:
    #     return (n+0.5)*log(n) - n + 0.5*log(2*pi) #n*log(n) - n + 0.5*log(2*pi*n)

def stirling(n):
    return (n+0.5)*log(n) - n + 0.5*log(2*pi)
    
np_log_fac = lambda xs:gammaln(xs+1)#np.vectorize(log_fac)

def timestamp(x):
    print ctime()
    return x
