import sys
sys.path.append("../data/chip_seq_datasets")
from viz_map import read_map,plot_reads,density_from_reads
from utils import separate,concat,mean
import random
from chip_seq import chip,alo
from numpy import random as nprand
from matplotlib import pyplot as plt
from math import log,exp
from utils import product,mh
    
def p_first_occ_at_i(strand,start,stop,hyp,i):
    if strand == '+':
        return hyp[i%G]*product((1-hyp[j%G]) for j in xrange(start,i,+1))
    else:
        return hyp[i%G]*product((1-hyp[j%G]) for j in xrange(stop,i,-1))

def p_frag_extends_to(strand,start,stop,i,lamb):
    if start <= i % G < stop:
        return 1
    else:
        endpoint = stop if strand == '+' else start
        d = abs(i - endpoint)
        return (1-lamb)**d
        
def p_occ_in_seq_region(strand,start,stop,hyp):
    return alo(hyp[start:stop])
    
def log_likelihood(reads,hyp,lamb=1/250.0,G=10000):
    ll = 0
    mfl = int(1/lamb)
    for (strand,start,stop) in reads:
        # Condition on the first occupancy at base i
        i_list = xrange(start,stop+10*mfl,+1) if strand == '+' else xrange(stop,start-10*mfl,-1)
        #likelihood = sum(p_first_occ_at_i(strand,start,stop,hyp,i)*p_frag_extends_to(strand,start,stop,i,lamb)
        #                 for i in i_list)
        likelihood = compute_likelihood_spec2(strand,start,stop,hyp,lamb)
        #print strand,start,stop,likelihood
        ll += log(likelihood)
    return ll

def compute_likelihood_ref(strand,start,stop,hyp,lamb):
    i_list = xrange(start,stop+10*mfl,+1) if strand == '+' else xrange(stop,start-10*mfl,-1)
    likelihood = sum(p_first_occ_at_i(strand,start,stop,hyp,i)*p_frag_extends_to(strand,start,stop,i,lamb)
                     for i in i_list)
    return likelihood

def compute_likelihood_spec(strand,start,stop,hyp,lamb):
    likelihood = 0
    i_list = xrange(start,stop+10*mfl,+1) if strand == '+' else xrange(stop,start-10*mfl,-1)
    p_no_occ_so_far = 1
    for i in i_list:
        likelihood += p_no_occ_so_far*p_frag_extends_to(strand,start,stop,i,lamb)*hyp[i%G]
        p_no_occ_so_far *= (1-hyp[i%G])
    return likelihood

def compute_likelihood_spec2(strand,start,stop,hyp,lamb):
    i_list = xrange(stop,stop+10*mfl,+1) if strand == '+' else xrange(start,start-10*mfl,-1)
    likelihood = alo(hyp[start:stop])
    p_no_occ_so_far = 1 - likelihood
    endpoint = stop if strand == '+' else start
    for i in i_list:
        likelihood += p_no_occ_so_far*(1-lamb)**abs(i-endpoint)*hyp[i%G]
        p_no_occ_so_far *= (1-hyp[i%G])
    return likelihood

def recovery():
    G = 10000
    config = [G/2]
    mfl = 250
    lamb = 1/float(mfl)
    num_frags = 1000
    frags = concat([chip(G,config,mfl) for i in xrange(num_frags)])
    min_seq_length = 75
    sequenced_frags = filter(lambda (start,stop):stop - start > min_seq_length,frags)
    fd_frags,bk_frags = separate(lambda x:random.random() < 0.5,sequenced_frags)
    fd_reads = [('+',start,start+75) for (start,stop) in fd_frags]
    bk_reads = [('-',stop-75,stop) for (start,stop) in bk_frags]
    reads = fd_reads + bk_reads
    hyp0 = [int(random.random() < 0.5) for i in range(G)]
    def f(hyp):
        return log_likelihood(reads,hyp,lamb,G)
    def prop(hyp):
        i = random.randrange(G)
        hyp_copy = hyp[:]
        hyp_copy[i] = 1 - hyp_copy[i]
        return hyp_copy
    chain = mh(f,prop,hyp0,use_log=True,verbose=True)
    
