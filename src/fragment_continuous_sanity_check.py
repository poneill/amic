import random
from utils import pairs,concat

def breaks(lamb):
    """break unit stick continuously at rate lambda (i.e. mean frag length = 1/lamb)"""
    x = 0
    bs = []
    while x < 1:
        bs.append(x)
        x += random.expovariate(lamb)
    bs.append(1)
    return bs

def make_frags(lamb):
    return pairs(breaks(lamb))

def frag_lengths(frags):
    return [(stop-start) for (start,stop) in frags]

def make_frag_lengths(lamb,trials):
    return frag_lengths(concat([make_frags(lamb) for trial in range(trials)]))

def frag_length(frags,i):
    for start,stop in frags:
        if start <= i < stop:
            return stop - start
    print i,frags
    assert False

def make_frag_length_i(lamb,p,trials):
    return [frag_length(make_frags(lamb),p) for i in range(trials)]
        
def simulate_frag_length_i(lamb,trials):
    return [random.expovariate(lamb) + random.expovariate(lamb) for i in range(trials)]
