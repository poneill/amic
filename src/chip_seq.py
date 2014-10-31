"""
Functions for generating chip seq datasets
"""
import random
from project_utils import *
#from simulate_tf import theoretical_probabilities
from utils import verbose_gen,pairs,concat,choose,choose2,mh
import numpy as np
from numpy import random as nprand
from scipy import sparse
from scipy.sparse import linalg
from combinatorial_utils import log_choose,log_fac
from fd import rfd_xs,rfd_xs_np

# def chip_seq_fragments(energy_matrix,genome,num_fragments):
#     """Simulate a chIP-seq dataset for energy_matrix on genome containing
#     given number of fragments.  Return a list of tuples containing
#     (left_endpoint,right_endpoint) of fragments,
#     """
#     fragments = []
#     print "computing theoretical probabilities"
#     true_probs = theoretical_probabilities(energy_matrix,genome)
#     sampler = inverse_cdf_sampler(true_probs)
#     for i in verbose_gen(xrange(num_fragments)):
#         # Determine position of TF on genome
#         pos = sampler()
#         # Determine length of fragment containing TF
#         frag_length = rpois(MEAN_FRAG_LENGTH)
#         # Determine distance from TF to left-endpoint of fragment
#         offset = min(random.randrange(frag_length),pos)
#         # collect left, right endpoints of fragment in list of fragments
#         fragments.append((pos-offset,pos-offset+frag_length))
#     return fragments

def make_splits_ref(G,lamb):
    splits = [0] + [i for i in xrange(G) if random.random() < lamb] + [G]# could be made more efficient by geometric r.v.'s...

def make_splits(G,lamb):
    splits = []
    i = 0
    while i < G:
        splits.append(i)
        i += nprand.geometric(lamb)
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
        i = bisect.bisect(splits,pos)
        bound_fragments.append((splits[i-1],splits[i]))
    return list(set(bound_fragments))

def chip_spec_deprecated(G,config,mean_frag_length):
    lamb = 1.0/mean_frag_length
    raw_frags = [(max(0,i-nprand.geometric(lamb)),min(G,i+nprand.geometric(lamb))) for i in config]

def chip_ps_np(ps,mean_frag_length,cells=10000,verbose=False):
    """Do a chip seq experiment given the distribution ps"""
    w = 10
    G = len(ps)# + w - 1 #XXX HACK
    cell_iterator = verbose_gen(xrange(cells),modulus=1000) if verbose else xrange(cells)
    return concat(chip(G,rfd_xs_np(ps),mean_frag_length) for cell in cell_iterator)

def reads_from_ps(ps,mfl,min_seq_len,num_reads):
    """Given a vector of chromosomal occupancies and mean fragment length, return specified number of reads"""
    reads = []
    lamb = 1.0/mfl
    G = len(ps)
    sampler = inverse_cdf_sampler(ps)
    while len(reads) < num_reads:
        L,R = nprand.geometric(lamb),nprand.geometric(lamb) #can be optimized to gamma(2,lamb)
        if L + R < min_seq_len:
            continue
        i = sampler()
        start,stop = (i - L) % G, (i + R)%G
        strand = "+" if random.random() < 0.5 else "-"
        if strand == "+":
            reads.append((strand,start,(start+min_seq_len)%G))
        else:
            reads.append((strand,(stop-min_seq_len)%G,stop))
    return reads

def density_from_ps(ps,mfl,min_seq_len,num_reads):
    """

    This function combines reads_from_ps and density_from_reads.  For
    some reason, slightly slower than generating and mapping reads
    separately...

    """
    G = len(ps)
    lamb = 1.0/mfl
    fwd_map = np.zeros(G)
    rev_map = np.zeros(G)
    reads_so_far = 0
    sampler = inverse_cdf_sampler(ps)
    problematic_reads = 0
    while reads_so_far < num_reads:
        L,R = nprand.geometric(lamb),nprand.geometric(lamb) #can be optimized to gamma(2,lamb)
        if L + R < min_seq_len:
            continue
        reads_so_far += 1
        i = sampler()
        start,stop = (i - L) % G, (i + R)%G
        if random.random() < 0.5:
            strand = "+"
            stop = (start + min_seq_len) % G
        else:
            strand = "-"
            start = (stop - min_seq_len) % G
        if start < stop: #i.e. if read doesn't wrap around
            if strand == "+":
                fwd_map[start:stop] += 1
            else:
                rev_map[start:stop] += 1
        else:
            problematic_reads += 1
            if strand == "+":
                fwd_map[start:G] += 1
                fwd_map[0:stop] += 1
            else:
                rev_map[start:G] += 1
                rev_map[0:stop] += 1
    print "problematic reads:",problematic_reads
    return fwd_map,rev_map
            
def chip_ps_const_frag_length_ref(ps,mean_frag_length,cells=10000,verbose=False):
    G = len(ps)
    out = np.zeros(G)
    G_iterator = verbose_gen(xrange(G),modulus=1000) if verbose else xrange(G)
    for i in G_iterator:
        ps_i = ps[max(i-mean_frag_length/2,0):min(i+mean_frag_length,G)]
        out[i] = alo(ps_i)
    return cells*out
    
def chip_ps_const_frag_length_ref2(ps,mean_frag_length,cells=10000,verbose=False):
    G = len(ps)
    out = np.zeros(G)
    cell_iterator = verbose_gen(xrange(cells),modulus=10000) if verbose else xrange(cells)
    for cell in cell_iterator:
        i = random.randrange(G-mean_frag_length)
        left,right = max(i-cutoff,0),min(i+cutoff,G)
        ps_i = ps[left:right]
        if random.random() < alo(ps_i):
            out[left:right] += 1
    return out

def chip_ps_spec2(ps,mean_frag_length,cells=10000):
    """Replaces chip_ps_np.  Try anything at this point..."""
    G = len(ps)
    out = np.zeros(G)
    cutoff = mean_frag_length * 5
    for i in range(G):
        nbhd = ps[i-cutoff:i+cutoff]
        out[i] += cells*alo(nbhd)
    return out

def reads_from_chip_ps_np(fragments,min_seq_length):
    """Given a list of fragments, return a list of (randomly) oriented
    reads for fragments >= min_seq_length.  Assumes no chromosomal
    wrapping in fragments.

    """
    reads = []
    for (start,stop) in fragments:
        if stop - start < min_seq_length:
            continue
        strand = "+" if random.random() < 0.5 else "-"
        if strand == "+":
            reads.append((strand,start,start+min_seq_length))
        else:
            reads.append((strand,stop-min_seq_length,stop))
    return reads
    
def mapped_reads_from_chip_ps_np(ps,mean_frag_length,cells=10000):
    G = len(ps)
    genome = np.zeros(G)
    lamb = 1.0/mean_frag_length
    for cell in (xrange(cells)):
        print cell
        config = rfd_xs_np(ps)
        config_len = len(config)
        idx = 0
        l = 0
        r = 0
        while l < G and idx < config_len:
            #print l,r,idx,config[idx]
            l,r = r,r+nprand.geometric(lamb)
            if l <= config[idx] < r:
                #print "got hit"
                genome[l:r] += 1
                while idx < config_len and config[idx] < r:
                    idx += 1
    return genome

def chip_ps(ps,mean_frag_length,cells=10000):
    """Do a chip seq experiment given the distribution ps"""
    w = 10
    G = len(ps) + w -1
    return concat(chip(G,rfd_xs(ps),mean_frag_length) for cell in verbose_gen(xrange(cells)))

def chip_seq_ref(ps,mean_frag_length=250,cells=10000,verbose=False):
    G = len(ps)
    return map_reads(chip_ps_np(ps,mean_frag_length,cells,verbose=verbose),G)

def chip_seq_spec(ps,mean_frag_length=250):
    """do phi computations 'by hand'"""
    G = len(ps)
    out = np.zeros(G)
    for i in range(G):
        pass

def chip_ps_ref(ps,mean_frag_length,cells=10000):
    """Do a chip seq experiment given the distribution ps"""
    G = len(ps)
    return concat(chip_ps(rfd_xs(ps),mean_frag_length)
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
        for i in xrange(start,stop):
            if not 0<= i < G:
                print i,start,stop
            genome[i] += 1
    return genome

def map_reads_np_deprecated(reads,G):
    print "WARNING: use map_reads_np instead!"
    genome = np.zeros(G)
    for (start,stop) in reads:
        for i in xrange(start,stop):
            genome[i] += 1
    return genome

def map_reads_np(reads,G):
    #print "WARNING: use map_reads instead!"
    genome = np.zeros(G)
    print "len reads:",len(reads)
    for (start,stop) in reads:
        genome[start:stop] += 1
    return genome
    
def map_read_endpoints_np(reads,G,length=75):
    """Simulate sequencing process more realistically by reading only 75 bp in either direction"""
    genome = np.zeros(G)
    skipped_reads = 0
    for (start,stop) in reads:
        if stop - start < length:
            skipped_reads += 1
            continue
        else:
            for i in range(length):
                genome[start + i] += 1
                genome[stop - i] += 1
    print "skipped %1.2f percent of reads for insufficient length" % (skipped_reads/float(len(reads))*100)
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
    """return probability of at least one (independent) event occurring"""
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

def tent2(x,i,mean_frag_length):
    lamb = 1.0/mean_frag_length
    return (1-lamb)**abs(x-i)

def tent3(x,i,mean_frag_length):
    """tent function for x, centered at i with rate """
    eff_lamb = 1.0/(mean_frag_length) # empircally determined; worrisome
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
    print "template:",len(template)
    #ones = np.ones(G)
    for i,p in enumerate(ps):
        shift = i - cutoff
        rolled_template = p*np.roll(template,shift)
        #print "len(rolled_template):",len(rolled_template)
        out = 1 - (1-out)*(1-rolled_template)
    return cells*out

def mysign(x):
    if x != 0:
        return sign(x)
    else:
        return 1

def predict_chip_ps5(ps,mean_frag_length,cells=100):
    G = len(ps)
    lamb = 1.0/mean_frag_length
    cutoff = min(5*mean_frag_length,G) # ignore contributions outside 5 times expected fragment length
    ks = range(-cutoff,cutoff)
    def left(i):
        return sum(ps[j]*product(1-ps[k] for k in range(j+1,i+1))*(1-lamb)**abs(j-i)
                   for j in range(i) if abs(j-i) < cutoff)
    def right(i):
        return sum(ps[j]*product(1-ps[k] for k in range(i,j))*(1-lamb)**abs(j-i)
                   for j in range(i,G) if abs(j-i) < cutoff)
    # return [cells*sum(ps[j]*product(1-ps[k] for k in range(i,j,mysign(j-i)))*(1-lamb)**abs(j-i)
    #                   for j in range(G) if abs(j-i) < cutoff)
    #         for i in range(G)]
    return [cells*(1-(1-left(i))*(1-right(i))) for i in verbose_gen(range(G))]

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
    """Given hypothesis ps, a chip-seq dataset in the form of mapped
    reads, and total number of cells, compute log likelihood--
    reference implementation.  Note that pi is an hypothesis about the
    probability that a fragment covers base i, not an hypothesis that base i is occupied.
    """

    def log_dbinom(N,k,p):
        return log_choose(N,k) + k*log(p) + (N-k)*log(1-p)

    return sum([log_dbinom(N,m,p) for m,p in verbose_gen(zip(mapped_reads,ps),modulus=1000)])

def chip_seq_log_likelihood(pds,tds,N):
    """Given a predicted chip seq dataset pds, the true chip-seq dataset
    tds (both in the form of mapped reads), and total number of cells in the original experiment,
    compute log likelihood.

    """
    # print "len(pred):",len(pred)
    # print "len(mapped_reads):",len(mapped_reads)
    #pred = np.minimum(np.array(pred)/N,1-1e-10)
    #print len(pds),len(tds)
    if not type(pds) is np.ndarray:
        print "convertings pds to numpy array"
        pds = np.array(pds)
    if not type(tds) is np.ndarray:
        print "convertings tds to numpy array"
        tds = np.array(tds)
    #pred = (pds + 1)/(N+1) # add pseudocounts of 0 and 1 # moved this to fd_inference
    ans = np.sum(log_fac(N) - (log_fac(tds) + log_fac(N-tds)) + tds*np.log(pds) + (N-tds)*np.log(1-pds))
    if ans == float('inf') or ans == float('-inf') or ans == float('nan'):
        print "ERROR in chip_seq_log_likelihood"
    return ans



def compare(G,endpoints,mean_frag_length,cells=10000,trials=10,semilogy=True):
    show_chip_shadow(G,endpoints,mean_frag_length,cells=cells,trials=trials)
    plt.plot(predict_chip_shadow(G,endpoints,mean_frag_length,cells=cells),color='r')
    if semilogy:
        plt.semilogy()
    plt.show()

def chip_ps_ising(ps,mean_frag_length,cells=10000,iterations=50000,x0=None,verbose=False):
    eps = -np.log(ps/(1-ps))
    lamb = 1.0/mean_frag_length
    coupling = -mean(eps) + log(lamb/(1-lamb))
    #coupling = -log(mean_frag_length)
    G = len(eps)
    if x0 is None:
        x0 = np.zeros(G)
    def hamiltonian(xs):
        field_contrib = np.dot(xs,eps)
        coupling_contrib = coupling*np.dot(np.diff(xs) == 0,xs[:-1])
        # if random.random() < 0.001:
        #     print "field contrib:",field_contrib,"coupling contrib:",coupling_contrib
        return field_contrib + coupling_contrib # bonus for [...,1,1,...]
        #return coupling/2.0*sum(np.diff(xs)) # penalty for differences
    def propose(xs):
        if random.random() < 0.001:
            print "occupation number:",np.sum(xs)
        ys = np.array(xs)
        i = random.randrange(G)
        ys[i] = 1 - ys[i]
        return ys
    def propose2(xs):
        return np.random.random(G) < ps
    def propose3(xs):
        flip_p = 1.0/mean_frag_length
        flip = 0
        ys = xs[:]
        for i in range(G):
            if random.random() < flip_p:
                flip = 1 - flip
            ys[i] = ys[i] - flip
        return ys
    def log_dprop2(xs,ys):
        return np.dot(np.log(ps),xs) + np.dot(np.log(1-ps),1-xs)
    chain = mh(f=lambda xs:-hamiltonian(xs),proposal=propose,iterations=iterations,
               x0=x0,dprop=None,use_log=True,verbose=verbose)
    return chain
        
