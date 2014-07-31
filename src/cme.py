"""
Solve the system using the full Chemical Master Equation.
"""
import numpy as np
import scipy
import scipy.linalg as la
import scipy.integrate as sint
import random,itertools
from utils import choose,hamming,find,transpose

def enumerate_states(G,q):
    def f(combo):
        return tuple([int(i in combo) for i in range(G)])
    return [f(combo) for i in range(q+1) for combo in itertools.combinations(range(G),i)]
    
def rate_matrix(q,koffs,verbose=False):
    """Generate the stochastic rate matrix for the givens system."""
    # Chromosome states can be represented by binary numerals; order the
    # states this way.
    G = len(koffs)
    states = enumerate_states(G,q)
    num_states = len(states)
    assert len(states) == sum(choose(G,i) for i in range(q+1))
    R = np.zeros((num_states,num_states))
    for i,state_i in enumerate(states):
        for j,state_j in enumerate(states):
            if verbose:
                print "considering:",i,state_i,"->",j,state_j
            dist = hamming(state_i,state_j)
            if dist != 1:
                # deal with diagonal elements later...
                if verbose:
                    print "distance is:",dist,"continuing..."
                continue
            if sum(state_j) == sum(state_i) + 1:
                R[i][j] = q - sum(state_i)
                if verbose:
                    print i,state_i,"->",j,state_j, "is an on-reaction, rate:",R[i][j]
            elif sum(state_j) == sum(state_i) - 1:
                diff_idx,diff_site = find(lambda (idx,(si,sj)):si != sj,enumerate(zip(state_i,state_j)))
                R[i][j] = koffs[diff_idx]
                if verbose:
                    print i,state_i,"->",j,state_j, "is an off-reaction (at site",diff_idx,")  rate:",R[i][j]
    # deal with diagonal elements
    for i in range(num_states):
        R[i][i] = -sum(R[i])
    print "finished rate matrix"
    return R
            
def solve(q,koffs,eps=1e-15,t_step=1):
    G = len(koffs)
    A = rate_matrix(q,koffs)
    num_states = len(A)
    def yprime(y,t):
        return y.dot(A)
    y0 = np.array([1] + [0]*(len(A)-1))
    t = np.linspace(0,t_step,2)
    y = sint.odeint(yprime,y0,t)[-1]
    diff = (np.linalg.norm(y-y0))
    while diff > eps:
        y0 = y
        y = sint.odeint(yprime,y0,t)[-1]
        diff = (np.linalg.norm(y-y0))
        print diff
    return marginalize(G,q,y)

def marginalize(G,q,y):
    """Given a vector y describing probabilities of being in any given
    state of the CME for (G,q), compute marginal probabilities"""
    states = enumerate_states(G,q)
    occs = np.zeros(G)
    assert len(y) == len(states)
    for yi,state in zip(y,states):
        occs += np.array(state)*yi
    return occs

def solve_rate_matrix(A,eps=1e-15):
    v =  null(np.transpose(A),eps=eps)
    return v/sum(v)
    
def null(A, eps=1e-15):
    """
    Compute nullspace of A.  Thanks Robert Kern and Ryan Krauss:
    http://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    """
    u, s, vh = la.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)
