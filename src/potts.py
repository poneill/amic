"""
Here we consider multiple copy system as a G+1-state Potts Model and
find the associated mean field approximation.
"""
from math import exp,log
from utils import product,choose2,l2,choose

beta = 1.0
G = 2
ks = [1,1]#[exp(-beta*random.gauss(-2,1)) for i in xrange(G)]
Q = [1]+ks # set of values each component of x-vector can take on.
eps = [-log(k) for k in Q]

# Q[0] is the off state, Q[i], i>0 represents binding to site i.
J = 100 # copy num.

### Notation
# i ranges over sites 0 (off-site) + [1..G]
# j ranges over copies [0,J).
# xj, the value of the jth copy, takes values in Q = [0,G] 
exclusion_penalty = 10**-100

def psi_j(xj):
    return Q[xj]

def psi_jjp(xj,xjp):
    if xj == xp > 0:
        return exclusion_penalty
    else:
        return 1
    
def Psi(xs):
    """return probability weight associated with configuration, up to
    Z.  P(xs) = Psi(xs)/Z"""
    single_terms = product(psi_jjp(x) for x in xs)
    pair_terms = product(psi_jjp(xj,xjp) for (xj,xjp) in choose2(xs))
    return  single_terms * pair_terms 
    
# def mean_field_nus():
#     """compute mean field approximation Q(xs) = \prod\nu_j(xj).
#     See Eq.4.2.9 of: http://www.stanford.edu/~montanar/TEACHING/Stat375/handouts/notes_stat375_1.pdf """
#     nus = [lambda x:1 for j in range(copy_num)]
#     iterations = 100
#     for iteration in iterations:
#         nus = [lambda xi:psi_jjp(xi)*exp(sum( for j,xj in enumerate())) for i in range(copy_num)]

def mean_field_hs():
    """Following derviation on wikipedia's mean field theory page..."""
    def V(xj,xjp):
        if xj == xjp > 0:
            retval = 10**10
        else:
            retval = (eps[xj] + eps[xjp])/choose(J,2) # divide by choose(J,2) since we're summing over pairs
        if random.random() < 0:
            print "V(%s,%s) = %s" % (xj,xjp,retval)
        return retval
    # because each term appears J-1 times.  self-consistency equation
    # for mean field approximation is:

    # Pj(xj) = 1/Z0 *exp(-beta*hj(xj)), where

    # hj(xj) = \sum_{<j,jp>} \sum_{xjp \in jp} V(xj,xjp)*Pjp(xjp)

    # In this case, the graph is fully connected and all variables xj
    # are exchangeable, so sum over pairs reduces to (J-1).

    # Moreover, due to exchangeability hj is the same for each
    # variable, so we can update a single function h(x) for all
    # variables.  h(x) is a function with G+1 possible input values,
    # so we can represent h as an array of size G+1 such that h[i]
    # stores the value h(i).

    # Initialize it arbitrarily
    h_cur = [1] * (G+1)
    h_next = [0] * (G+1)
    def P(i):
        """Return probability at time t that x takes on value i"""
        return exp(-beta*h_cur[i])/sum(exp(-beta*h_cur[ip]) for ip in range(G+1))
    while True:
        for i in range(G+1):
            terms = [V(i,ip)*P(ip) for ip in range(G+1) if not i == ip]
            #print i,terms
            h_next[i] = (J-1)*sum(terms)
        if l2(h_next,h_cur) < 10**-10:
            break
        h_cur = h_next[:]
        print h_cur
    return h_cur
        
def probs_from_mean_field_hs(mf_hs):
    """compute occupancies, given mean_field_hs"""
    Z = sum(exp(-beta*mfh) for mfh in mf_hs)
    return [exp(-beta*mfh)/Z for mfh in mf_hs]
