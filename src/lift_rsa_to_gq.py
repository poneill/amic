"""
In this script we attempt to sample from the Gq model by using Random
Sequential Adsorption (RSA) as a proposal function.
"""
from sample import rsa
from utils import mh,fac
import random
from project_utils import falling_fac
from math import exp,log
import time

def main():
    sigma = 8
    ks = [1] + [exp(random.gauss(0,sigma)) for i in range(100)] #k0 is off-state
    G = len(ks)-1
    q = 5
    def Pstar(xs):
        """Compute probability of config under Gq model, up to Z"""
        weight = falling_fac(q,len([x for x in xs if x > 0]))
        return weight * product([ks[x] for x in xs])
    def rQ(xs):
        """given current configuration, sample one independently using rsa"""
        return smart_rsa(ks,q)
    def dQ(xs,xs_last):
        """Return probability of configuration under rsa"""
        _ks = ks[:]
        prob = 1
        for x in xs:
            k = _ks[x]
            prob *= k/sum(_ks)
            if x > 0:
                _ks[x] = 0
        return prob
    tic = time.time()
    chain = mh(Pstar,rQ,[0,0,0,0,0],dQ)
    toc = time.time()
    print "ran chain in:",toc-tic
    print "starting direct sampling"
    tic = time.time()
    test_xs = [direct_sampling(ks,q) for i in verbose_gen(xrange(50001),
                                                          modulus=1)]
    toc = time.time()
    print "direct sampling in:",toc-tic
    ss = [ss_from_xs(xs,G) for xs in chain]
    test_ss = [ss_from_xs(xs,G) for xs in test_xs]
    plt.plot(map(mean,transpose(ss)),label="Lifting")
    plt.plot(map(mean,transpose(test_ss)),label="Direct Sampling")
    plt.xlabel("Chromosomal coordinate")
    plt.ylabel("Occupancy")
    plt.legend()
    plt.show()
