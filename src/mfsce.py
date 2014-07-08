"""
Mean-field self-consistency equations

Hp(x) = sum_<xj,xjp>(psi(xj,xjp))
V(xj,xjp) = exp(-beta*psi(xj,xjp))
Hq(x) = sum_xj hj(xj)

Pj(xj) = 1/Z0 *exp(-beta*hj(xj)), where
hj(xj) = \sum_{<j,jp>} \sum_{xjp \in jp} V(xj,xjp)*Pjp(xjp)

"""
beta = 1
from math import exp,log
from matplotlib import pyplot as plt
import random
import itertools
from scipy.stats import pearsonr
from utils import pairs,log2,concat,l2,mh,inverse_cdf_sample


def hamilp(xs):
    def psi12(x1,x2):
        return x1*x2
    def psi13(x1,x3):
        return x1*x3 - x2
    def psi23(x2,x3):
        return x2*x3 + x3
    x1,x2,x3 = xs
    return psi12(x1,x2) + psi13(x1,x3) + psi23(x2,x3)

states = [(x1,x2,x3) for x1 in range(2)
          for x2 in range(2)
          for x3 in range(2)]

Zp = sum(exp(-beta*hamilp(xs)) for xs in states)

def P(xs):
    return exp(-beta*(hamilp(xs)))/Zp

def mean_field_hs(Vs,K):
    """

    Pj(xj) = 1/Z0 *exp(-beta*hj(xj)), where
    hj(xj) = \sum_{<j,jp>} \sum_{xjp \in jp} V(xj,xjp)*Pjp(xjp)

    We assume a Potts model of m variables x0...xj...xm-1 where each
    variable can take on K states 0...i...K-1.  Mean field functions h
    are represented as a matrix hss where each row gives the values
    hj(i).  [Note that i,j are reversed from the usual row-column
    convention.]

    Input is a matrix Vs of pairwise contributions to the hamiltonian
    where Vs[j][jp] is a function V(xj,xjp)
    """
    M = len(Vs)
    jpairs = pairs(range(M))
    hs = [[1 for i in range(K)] for j in range(M)]
    def Pj(xj,j):
        #print xj,j
        return exp(-beta*hs[j][xj])/sum(exp(-beta*hs[j][xjp]) for xjp in range(K))
    old_hs = matcopy(hs)
    while True:
        for j in range(M):
            for i in range(K):
                hs[j][i] = (sum(sum(Vs[j][jp](i,ip)*Pj(ip,jp)
                                       for ip in range(K))
                                    for jp in range(j+1,M)) +
                                sum(sum(Vs[jp][j](ip,i)*Pj(ip,jp)
                                       for ip in range(K))
                                    for jp in range(0,j-1)))
        print l2(concat(hs),concat(old_hs))
        if old_hs == hs:
            break
        else:
            old_hs = matcopy(hs)
            print hs
    return hs

def matcopy(xxs):
    return [xs[:] for xs in xxs]

def mean_field_test(M=10,K=2,sigma=1,plotting=True):
    Vs = [[None for j in range(M)] for jp in range(M)]
    for j in range(M):
        for jp in range(j+1,M):
            d = {(xj,xjp):random.gauss(0,sigma)
                 for xj in range(K) for xjp in range(K)}
            Vjjp = lambda xj,xjp:d[(xj,xjp)]
            Vs[j][jp] = Vjjp
    states = list(itertools.product(*[range(K) for j in range(M)]))
    def Hp(xs):
        return sum(Vs[j][jp](xj,xjp)
                   for ((j,xj),(jp,xjp))
                   in itertools.combinations(enumerate(xs),2))
    mf_hs = mean_field_hs(Vs,K)
    print "computing Zp"
    Zp = sum(exp(-beta*Hp(xs)) for xs in states)
    def P(xs):
        return exp(-beta*Hp(xs))/Zp
    def Hq(xs):
        return sum(mf_hs[j][xj] for j,xj in enumerate(xs))
    print "computing Zq"
    Zq = sum(exp(-beta*Hq(xs)) for xs in states)
    def Q(xs):
        return exp(-beta*Hq(xs))/Zq
    # for state in states:
    #     print state,P(state),Q(state)
    ps = [P(state) for state in states]
    qs = [Q(state) for state in states]
    print pearsonr(ps,qs)
    print "Sp (bits):",sum(-p*log2(p) for p in ps)
    print "Sq (bits):",sum(-q*log2(q) for q in qs)
    print "Dkl(P||Q) (bits):",sum(p*log2(p/q) for p,q in zip(ps,qs))
    def rQ(xs):
        """MFA proposal"""
        return [inverse_cdf_sample(range(K),boltzmann(mf_h)) for mf_h in mf_hs]
    def rR(xs):
        """Uniform proposal"""
        return [random.choice(range(K)) for j in range(M)]
    mh(f=P,proposal=rQ,dprop=Q,x0=[0]*M)
    mh(f=P,proposal=rR,x0=[0]*M)


    
    if plotting:
        plt.scatter(ps,qs)
        plt.xlabel("Ps")
        plt.ylabel("Qs")
        plt.loglog()
        minp,maxp = min(ps),max(ps)
        minq,maxq = min(qs),max(qs)
        plt.plot([minp,maxp],[minq,maxq])
        plt.xlim(minp,maxp)
        plt.ylim(minq,maxq)
        plt.show()

def boltzmann(eps):
    """Compute boltzmann distribution given hamiltonian eps"""
    Z = sum(exp(-beta*ep) for ep in eps)
    return [exp(-beta*ep)/Z for ep in eps]
