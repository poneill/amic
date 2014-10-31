"""
chip seq datasets come in the form of forward, backward reads.  We
need to convert these to fragments.  From a set of fwd and bkd reads,
reconstruct a set of fragments.
"""
import sys
sys.path.append("../data/chip_seq_datasets")
from viz_map import read_map,plot_reads,density_from_reads
from utils import separate,concat,mean,verbose_gen
import random
from chip_seq import chip
from numpy import random as nprand
from matplotlib import pyplot as plt

def frags_from_map(filename,N,mfl=250):
    reads = read_map(filename)
    fragments = infer_frags(reads,N,mfl)
    return fragments
        
def infer_frags(reads,N,mfl=250.0*2,return_acceptance_rate=False):
    mfl = float(mfl)
    fwds,bcks = separate(lambda (strand,start,stop):strand=='+',reads)
    fragments = []
    attempts = 0
    while len(fragments) < N:
        _, fwd_start, fwd_stop = random.choice(fwds)
        _, bck_start, bck_stop = random.choice(bcks)
        distance = bck_start - fwd_stop
        attempts += 1
        if distance < 0:
            continue
        accept_p = exp(-distance/mfl)
        if random.random() < accept_p:
            fragments.append((fwd_start,bck_stop))
            if True:#len(fragments) % 5 == 0 and len(fragments) > 0:
                print "accepted:",-fwd_start+bck_stop,len(fragments),len(fragments)/float(attempts)
    if return_acceptance_rate:
        return N/float(attempts)
    else:
        return fragments

def infer_frags2(reads,mfl,extension=2):
    """Infer fragments as posterior hypotheses in a bayesian setting:

    P(frags|reads) = P(reads|frags)P(frags)/P(reads)
                   = 1*(\prod lambda^2(1-lambda)^(ri-li))/P(reads)"""
    lamb = 1/float(mfl)
    fwds,bcks = separate(lambda (strand,start,stop):strand=='+',reads)
    fragments = []
    alpha = sum(min_seq_length/float(min_seq_length+ell)*lamb*exp(-lamb*ell)
                for ell in xrange(100000))
    alpha = 0.5
    for (strand,start,stop) in reads:
        ext_length = sample_ext_length(lamb,min_seq_length,alpha)
        if strand == '+':
            frag = (start,stop + ext_length)
        else:
            frag = (start - ext_length,stop)
        fragments.append(frag)
    return fragments

def sample_ext_length(lamb=1/250.0,min_seq_length=75,alpha=None):
    """Given a read that extends for (at least) min_seq_length, sample the
distance it extends past the end of the sequenced read.

    If the position of the TF is inside the sequenced portion, then
    the endpoint of the fragment is distributed as Exp(lamb).  If the
    position of the TF is outside the sequenced portion, however, then
    the endpoint is distributed as Exp(lamb) + Exp(lamb) =
    Gamma(lamb,2).  We therefore model the distribution of the
    fragment endpoint as a mixture model:

    endpoint ~ alpha*Exp(lamb) + (1-alpha)*Gamma(lamb,2)

    where the mixing parameter alpha is the probability that the TF
    resides within the sequenced portion.  We assume that this probability can be described as:

    <sequenced_length/total_length>,

    where the expectation is taken with respect to the total length of
    the fragment, which is distributed exponentially.
    """
    if alpha is None:
        alpha = sum(min_seq_length/float(min_seq_length+ell)*lamb*exp(-lamb*ell) for ell in xrange(100000))
        alpha = 0.3
    if random.random() < alpha:
        # TF resides in binding_site
        ext_length = nprand.geometric(lamb)
    else:
        ext_length = nprand.geometric(lamb) + nprand.geometric(lamb)
    return ext_length
    
def test_frags_from_map():
    filename = "../data/chip_seq_datasets/ArcA_park_et_al/SRR835423/SRR835423.map"
    reads = read_map(filename)
    num_frags = 10000
    frags = frags_from_map(filename,num_frags,mfl=250)
    plot_reads(reads)
    plot_reads(map(lambda (start,stop):('+',start,stop),frags))

def frag_sabot(frags):
    """Convert frags to positive strand reads for plotting purposes"""
    return map(lambda (start,stop):('+',start,stop),frags)

def frag_density(frags,G):
    fwd_map,rev_map = density_from_reads(frag_sabot(frags),G=G)
    return fwd_map

def exp_reconstruction(reads,lamb,G):
    """Reconstruct fragment density map by assuming exponential extension of each read"""
    frag_map = [0]*G
    mfl = int(1/lamb)
    for (strand,start,stop) in verbose_gen(reads,modulus=10000):
        assert(stop - start == 75)
        for i in range(start,stop):
            frag_map[i] += 1
        ext_list = xrange(stop,stop+10*mfl,+1) if strand == "+" else xrange(start-10*mfl,start,+1)
        endpoint = stop if strand == "+" else start
        for i in ext_list:
            frag_map[i%G] += (1-lamb)**abs(i-endpoint)
    return frag_map
    
def sanity_check():
    G = 10000
    config = [G/2]
    mfl = 250
    lamb = 1.0/mfl
    num_frags = 10000
    frags = concat([chip(G,config,mfl) for i in xrange(num_frags)])
    min_seq_length = 75
    sequenced_frags = filter(lambda (start,stop):stop - start > min_seq_length,frags)
    fd_frags,bk_frags = separate(lambda x:random.random() < 0.5,sequenced_frags)
    fd_reads = [('+',start,start+min_seq_length) for (start,stop) in fd_frags]
    bk_reads = [('-',stop-min_seq_length,stop) for (start,stop) in bk_frags]
    reads = fd_reads + bk_reads
    inferred_frags = exp_reconstruction(reads,lamb,G)
    plot_reads(reads,G=G)
    plt.plot(frag_density(frags,G=G),label="all frags")
    plt.plot(frag_density(sequenced_frags,G=G),label="seq frags")
    plt.plot((inferred_frags),label="inferred frags")
    plt.legend()
