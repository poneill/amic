from sympy import *

def a(L,m):
    """
    Return number of arrangements of an unconserved (i.e. arbitrarily
    many) molecules of length m on lattice of length L, without
    counting permutations of molecules.
    """
    if L < m: # only trivial, empty lattice
        return 1
    else:
        return a(L-1,m) + a(L-m,m)

def b(L,n,m,debug=False):
    """Count number of arrangements of exactly n molecules of width m
    on lattice of length L, without counting permutations"""
    if n == 0:
        return 1
    elif L < m:
            return 0
    else:
        retval = b(L-1,n,m,debug=debug) + b(L-m,n-1,m,debug=debug)
        if debug:
            print "b(%s,%s,%s) = %s" %(L,n,m,retval)
        return retval

def c(L,n,m):
    """Count number of arrangements of up to n molecules of width m
    on lattice of length L, without counting permutations"""
    return sum(b(L,i,m) for i in range(n+1))
    
def find_a(L,m):
    x = sympy.var('x')
    G = 1/(1-x-x**m)
    return diff(G,x,L).subs(x,1)
