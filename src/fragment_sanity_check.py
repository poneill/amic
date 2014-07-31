"""
Get straight on statistics for fragment generation
"""
import random
from utils import pairs
from matplotlib import pyplot as plt
from math import log

G = 1000
genome = range(G)

def breaks(lamb,G):
    """1 denotes a break immediately after base i"""
    return [0] + [int(random.random() < lamb) for i in range(G-1)]

def fragments_from_breaks(breaks,G):
    endpoints = [0] + [i for (i,b) in enumerate(breaks) if b] + [G]
    return pairs(endpoints)

def len_fragment_covering_i(fragments,i):
    return [stop - start for (start,stop) in fragments if start<= i < stop][0]

def fragment_lengths(fragments):
    return [(stop - start) for (start,stop) in fragments]

def make_fragments(lamb,G):
    return fragments_from_breaks(breaks(lamb,G),G)
    
def rgeom(lamb):
    x = 1
    while(random.random() < 1-lamb):
        x += 1
    return x

def rgeom_fast(lamb):
    return int(log(random.random())/log(1-lamb)) + 1
    
def rgeom2(lamb):
    return rgeom(lamb) + rgeom(lamb) -1

    
