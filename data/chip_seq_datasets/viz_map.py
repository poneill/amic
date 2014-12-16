"""Convert a map file to read density map"""

from matplotlib import pyplot as plt
from math import exp
import numpy as np
from utils import verbose_gen

def read_map(filename):
    with open(filename) as f:
        coords = [(line[1],int(line[3]),len(line[4])) for line in map(lambda l:l.split("\t"),f.readlines())]
    reads = [(x[0],x[1],x[1] + x[2]) for x in coords]
    return reads

def density_from_reads_ref(reads,G=None):
    """From a collection of reads, return forward and reverse density maps.  Assume circular chromosome"""
    if G is None:
        G = max((map(lambda(strand,start,stop):max(start+1,stop),reads)))
        print G
    fwd_map = np.zeros(G)
    rev_map = np.zeros(G)
    problematic_reads = []
    for strand,start,stop in reads:
        if stop < start:#abs(stop - start) > G/10: #ie. if read wraps around origin...
            problematic_reads.append((strand,start,stop))
            continue
        if strand == "+":
            fwd_map[start:stop] += 1
        else:
            rev_map[start:stop] += 1
    print "problematic reads:",len(problematic_reads)
    for (strand,start,stop) in problematic_reads:
        if strand == "+":
            fwd_map[start:G] += 1
            fwd_map[0:stop] += 1
        else:
            rev_map[start:G] += 1
            rev_map[0:stop] += 1
    return fwd_map,rev_map

def density_from_reads(reads,G=None):
    """compute read density maps (fwd,rev) from reads.  see density_from_reads_ref"""
    if G is None:
        G = max((map(lambda(strand,start,stop):max(start+1,stop),reads)))
        print G
    fwd_deltas = np.zeros(G)
    rev_deltas = np.zeros(G)
    fwd_prob_reads = 0
    rev_prob_reads = 0
    for (strand, start, stop) in reads:
        if start > stop:
            if strand == "+":
                fwd_prob_reads += 1
            else:
                rev_prob_reads += 1
        if strand == "+":
            fwd_deltas[start] += 1
            fwd_deltas[stop] -= 1
        else:
            rev_deltas[start] += 1
            rev_deltas[stop] -= 1
    print "problematic reads:",fwd_prob_reads + rev_prob_reads
    fwd_rdm = np.cumsum(fwd_deltas) + fwd_prob_reads
    rev_rdm = np.cumsum(rev_deltas) + rev_prob_reads
    return fwd_rdm,rev_rdm
        
        
def plot_reads(reads,G=None):
    fwd_map, rev_map = density_from_reads(reads,G)
    comb_reads = [fd + rv for (fd,rv) in zip(fwd_map,rev_map)]
    plt.plot(fwd_map)
    plt.plot(rev_map)
    plt.plot(comb_reads)
    plt.show()

def plot_reads_explicit(reads):
    """Plot reads as overlapping line segments"""
    fwd_hist = [0]*5000000
    bck_hist = [0]*5000000
    for strand,start,stop in verbose_gen(reads,modulus=1000):
        if strand == '+':
            height = max(fwd_hist[start:stop]) + 1
            for i in range(start,stop,1):
                fwd_hist[i] += 1
            plt.plot([start,stop],[height]*2,color='b')
        else:
            height = max(bck_hist[stop:start]) + 1
            for i in range(start,stop,-1):
                fwd_hist[i] += 1
            plt.plot([start,stop],[-height]*2,color='g')
