from math import exp
import random
import numpy as np
from utils import verbose_gen
N = int(10**6)
Q = 10

if not 'scores' in dir():
    print "defining scores"
    scores = [exp(-random.random()) for i in range(N)]


def inner_Z(n,q):
    """Compute partition function for first n sites (up to and including
    position n in python list notation) with q particles.

    """
    #print "inner_Z(%s,%s)" % (n,q)
    if q == 0 or n < 0:
        return 1
    else:
        return Z(n-1,q) + scores[n]*Z(n,q-1)

arr = np.zeros((N,Q+1))

def Z(n,q):
    #print "Z(%s,%s)" % (n,q)
    if arr[n,q] == 0:
        arr[n,q] = inner_Z(n,q)
    return arr[n,q]

def initialize_array():
    for i in verbose_gen(xrange(N),modulus=1000):
        Z(i,Q)
        
