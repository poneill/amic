"""
Functions for generating chip seq datasets
"""
import random
from project_utils import *
from data import *
from simulate_tf import theoretical_probabilities
from utils import verbose_gen,pairs,concat,choose,choose2
import numpy as np
from numpy import random as nprandom
from scipy import sparse
from scipy.sparse import linalg
from combinatorial_utils import log_choose,log_fac
from fd import rfd_xs

def chip_seq_fragments(energy_matrix,genome,num_fragments):
    """Simulate a chIP-seq dataset for energy_matrix on genome containing
    given number of fragments.  Return a list of tuples containing
    (left_endpoint,right_endpoint) of fragments,
    """
    fragments = []
    print "computing theoretical probabilities"
    true_probs = theoretical_probabilities(energy_matrix,genome)
    sampler = inverse_cdf_sampler(range(L-W+1),true_probs)
    for i in verbose_gen(xrange(num_fragments)):
        # Determine position of TF on genome
        pos = sampler()
        # Determine length of fragment containing TF
        frag_length = rpois(MEAN_FRAG_LENGTH)
        # Determine distance from TF to left-endpoint of fragment
        offset = min(random.randrange(frag_length),pos)
        # collect left, right endpoints of fragment in list of fragments
        fragments.append((pos-offset,pos-offset+frag_length))
    return fragments

def make_splits_ref(G,lamb):
    splits = [0] + [i for i in xrange(G) if random.random() < lamb] + [G]# could be made more efficient by geometric r.v.'s...

def make_splits(G,lamb):
    splits = []
    i = 0
    while i < G:
        splits.append(i)
        i += nprandom.geometric(lamb)
    splits.append(G)
    return splits
    
def chip_ref(G,config,mean_frag_length):
    """Given a genome length G, configuration (vector giving
    locations of left endpoints of TFs), and mean fragment length,
    return a collection of fragments (in form [left endpoint, right
    endpoint)) for one cell."""
    lamb = 1.0/mean_frag_length
    splits = make_splits(G,lamb)
    all_endpoints = pairs(splits)
    bound_fragments = [(lep,rep) for (lep,rep) in all_endpoints if any(lep <= pos < rep for pos in config)]
    return bound_fragments

def frags_from_splits_ref(config,splits):
    all_endpoints = pairs(splits)
    bound_fragments = [(lep,rep) for (lep,rep) in all_endpoints if any(lep <= pos < rep for pos in config)]
    return bound_fragments

def frags_from_splits(config,splits):
    bound_fragments = []
    for pos in config:
        i = bisect.bisect_left(splits,pos)
        bound_fragments.append((splits[i-1],splits[i]))
    return list(set(bound_fragments))

def test_frags_from_splits(G,mean_frag_length,trials):
    lamb = 1.0/mean_frag_length
    for trial in verbose_gen(range(trials)):
        config = [random.randrange(G) for i in range(100)]
        splits = make_splits(G,lamb)
        if set(frags_from_splits(config,splits)) == set(frags_from_splits_ref(config,splits)):
            continue
        else:
            return config,splits
            
def chip(G,config,mean_frag_length):
    """Given a genome length G, configuration (vector giving
    locations of left endpoints of TFs), and mean fragment length,
    return a collection of fragments (in form [left endpoint, right
    endpoint)) for one cell."""
    #config = sorted(config)
    lamb = 1.0/mean_frag_length
    splits = make_splits(G,lamb)
    bound_fragments = []
    for pos in config:
        i = bisect.bisect_left(splits,pos)
        bound_fragments.append((splits[i-1],splits[i]))
    return list(set(bound_fragments))

def chip_ps(ps,mean_frag_length,cells=10000):
    """Do a chip seq experiment given the distribution ps"""
    G = len(ps)
    return concat(chip(G,rfd_xs(ps),mean_frag_length) for cell in verbose_gen(xrange(cells)))

def chip_ps_ref(ps,mean_frag_length,cells=10000):
    """Do a chip seq experiment given the distribution ps"""
    G = len(ps)
    return concat(chip_ref(G,rfd_xs(ps),mean_frag_length)
                  for cell in verbose_gen(xrange(cells)))

def chip_ps_spec(ps,mean_frag_length,cells=10000):
    return concat(chip_ps_spec_single_cell(ps,mean_frag_length)
                  for i in verbose_gen(xrange(cells)))
    
def chip_ps_spec_single_cell(ps,mean_frag_length):
    G = len(ps)
    out = [0]*G
    i = 0
    lamb = 1.0/mean_frag_length
    bound_fragments = []
    last_right_endpoint = 0
    while i < G:
        if random.random() < ps[i]:
            right_ep = min(rgeom_fast(lamb) + i,G)
            left_ep = max(-rgeom_fast(lamb)+i,last_right_endpoint)
            bound_fragments.append((left_ep,right_ep))
            i = right_ep
            last_right_endpoint = i
        i += 1
    return bound_fragments
            
        
        

def map_reads(reads,G):
    genome = [0]*G
    for (start,stop) in reads:
        for i in range(start,stop):
            genome[i] += 1
    return genome
    
def conv_mat(G,lamb,sigma=0):
    f = lambda i,j:(1-lamb)**abs(i-j) + abs(sigma*random.gauss(0,1))
    return np.fromfunction(f,(G,G))

def deconvolve_reads(mapped_reads,mean_frag_length):
    G = len(mapped_reads)
    lamb = 1.0/mean_frag_length
    M = conv_mat(G,lamb,sigma=0)
    config = linalg.bicg(M,mapped_reads)
    return config

def show_chip_shadow(G,endpoints,mean_frag_length,cells=10000,trials=10):
    lamb = 1.0/mean_frag_length
    [plt.plot(map_reads(concat([chip(G,endpoints,mean_frag_length) for i in range(cells)]),G),color='b')
     for i in verbose_gen(range(trials))]

def alo(ps):
    """return probability of at least one event occurring"""
    return 1 - product(1-p for p in ps)

def alo2(ps):
    return (sum(ps) - sum(pi*pj for pi,pj in choose2(ps)))

def alo_rec(ps):
    if len(ps) == 1:
        return ps[0]
    else:
        return 1-(1-ps[0])*(1-alo_rec(ps[1:]))
        
def tent(x,i,mean_frag_length):
    """tent function for x, centered at i with rate """
    eff_lamb = 1.0/(mean_frag_length-0.5) # empircally determined; worrisome
    return exp(-eff_lamb*abs(x-i))

def gauss_tent(x,i,mean_frag_length):
    """tent function for x, centered at i with rate """
    eff_lamb = 1.0/(mean_frag_length-0.5) # empircally determined; worrisome
    return exp(-eff_lamb*abs(x-i)**2)

def predict_chip_shadow(G,endpoints,mean_frag_length,cells=10000):
    lamb = 1.0/mean_frag_length
    return [cells*alo(tent(x,i) for i in endpoints) for x in range(G)]

def predict_chip_ps(ps,mean_frag_length,cells=10000):
    G = len(ps)
    eff_lamb = 1.0/(mean_frag_length-0.5) # empircally determined; worrisome
    return [cells*alo([p*tent(x,i,mean_frag_length) for i,p in enumerate(ps)])
            for x in verbose_gen(range(G))]

def predict_chip_ps2(ps,mean_frag_length,cells=10000):
    G = len(ps)
    eff_lamb = 1.0/(mean_frag_length-0.5) # empircally determined; worrisome
    return [cells*alo2([p*tent(x,i,mean_frag_length) for i,p in enumerate(ps)])
            for x in verbose_gen(range(G))]

def predict_chip_ps4(ps,mean_frag_length,cells=100):
    G = len(ps)
    cutoff = min(10*mean_frag_length,G) # ignore contributions outside 10 times expected fragment length
    template = np.array([tent(x,0,mean_frag_length) for x in range(-cutoff,cutoff+1)] + [0]*(G-2*cutoff-1))
    out = np.zeros(G)
    for i,p in enumerate(ps):
        shift = i - cutoff
        rolled_template = p*np.roll(template,shift)
        #print "len(rolled_template):",len(rolled_template)
        out = 1 - (1-out)*(1-rolled_template)
    return cells*out
    
def chip_convolution_matrix(G,mean_frag_length):
    cutoff = min(10*mean_frag_length,G) # ignore contributions outside 10 times expected fragment length
    template = np.array([tent(x,0,mean_frag_length) for x in range(-cutoff/2,cutoff/2+1)])
    offsets = range(-cutoff/2,cutoff/2+1)
    return sparse.diags(template,offsets,shape=(G,G))
    
def predict_chip_ps3(ps,mean_frag_length,cells=10000):
    #print "len(ps):",len(ps)
    G = len(ps)
    cutoff = min(10*mean_frag_length,len(ps)/2) # ignore contributions outside 10 times expected fragment length
    #print "cutoff:",cutoff
    template = np.array([tent(x,0,mean_frag_length) for x in range(-cutoff,cutoff)])
    pred = np.convolve(ps,template,mode='same')
    #print "raw_pred:",len(pred)
    assert(len(pred) == len(ps))
    return cells*pred#[cutoff:-cutoff]
        
def chip_seq_log_likelihood_ref(ps,mapped_reads,N):
    """Given hypothesis ps, a chip-seq dataset in the form of mapped reads, and total number of cells,
    compute log likelihood-- reference implementation"""

    def log_dbinom(N,k,p):
        return log_choose(N,k) + k*log(p) + (N-k)*log(1-p)

    return sum([log_dbinom(N,m,p) for m,p in verbose_gen(zip(mapped_reads,ps),modulus=1000)])

def chip_seq_log_likelihood(pred,mapped_reads,N):
    """Given predicted counts pred, a chip-seq dataset in the form of
    mapped reads, and total number of cells, compute log likelihood--
    reference implementation
    """
    # print "len(pred):",len(pred)
    # print "len(mapped_reads):",len(mapped_reads)
    pred = np.minimum(np.array(pred)/N,1-1e-10)
    ms = np.array(mapped_reads)
    return np.sum(log_fac(N) - (log_fac(ms) + log_fac(N-ms)) + ms*np.log(pred) + (N-ms)*np.log(1-pred))


        
def compare(G,endpoints,mean_frag_length,cells=10000,trials=10,semilogy=True):
    show_chip_shadow(G,endpoints,mean_frag_length,cells=cells,trials=trials)
    plt.plot(predict_chip_shadow(G,endpoints,mean_frag_length,cells=cells),color='r')
    if semilogy:
        plt.semilogy()
    plt.show()
