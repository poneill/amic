"""
this is just a little sketch to prove certain transformations between
operations on arrays and operations on their cumulative sums
"""

import numpy as np

def addition():
    N = 10
    i = 5
    xs = np.arange(N)
    ys = np.cumsum(xs)
    xs[i] += 14
    ys[i:] += 14
    print np.all(ys == np.cumsum(xs))

def zero_element():
    N = 10
    i = 5
    xs = np.arange(N)
    ys = np.cumsum(xs)
    xs[i] = 0
    xi = ys[i] - ys[i-1]
    ys[i:] -= xi
    print np.all(ys == np.cumsum(xs))

def zero_range():
    N = 10
    i,j = 5,8
    xs = np.arange(N)
    ys = np.cumsum(xs)
    print "xs:",xs
    print "ys:",ys
    xs[i:j] = 0
    zs = np.cumsum(xs)
    a,b = ys[i-1],ys[j-1]
    print a,b
    ys[i:j] = a
    ys[j:] -= (b - a)
    print "xs:",xs
    print "ys:",ys
    print "zs:",np.cumsum(xs)
    print (ys == zs)
