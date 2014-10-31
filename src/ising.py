"""
scratch pad for RFIM
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from math import log,exp,tanh,atanh,sqrt
from utils import interpolate,mean,transpose,mh,pairs,dot,verbose_gen,converge2,show,sd
from itertools import product
import numpy as np

def count_aligned_xs(xs):
    return np.dot(np.diff(xs) == 0,xs[:-1])

def count_aligned_xs2(xs):
    return np.sum(xs[:-1] * xs[1:])

def test_count_aligned_xs(trials):
    for i in xrange(trials):
        xs = 1*(np.random.random(N) < 0.5)
        if count_aligned_xs(xs) != count_aligned_xs2(xs):
            return False
    return True
    
        
def hamil_xs(xs,hs,J):
    """
    Assume spins take on values in [0,1], and a contribution of J only
    if neighboring spins are [1,1].
    """
    return np.dot(hs,xs) + J*count_aligned_xs(xs)

def hamil_xs2(xs,hs,J):
    return np.dot(hs,xs) + J*count_aligned_xs2(xs)
    
def from_xs(xs):
    """convert xs [0,1] to sigmas [-1,1]"""
    return 2*xs - 1

def hamil_sigmas(sigmas,hs_sigma,J_sigma):
    return np.dot(hs_sigma,sigmas) + J_sigma*np.sum(sigmas[:-1] * sigmas[1:])

def hamil_test(sigmas,hs_x,J_x):
    N = len(sigmas)
    hs_sigma = hs_x/2.0 + J_x
    J_sigma = J_x/4.0
    const = J_x*N/4.0 + np.sum(hs_x)/2.0
    field_term = np.dot(sigmas,hs_sigma)
    coupling_term = J_sigma*np.sum(sigmas[:-1] * sigmas[1:])
    print "field term:",field_term
    print "J_x*N/2:",J_x*N/2.0
    print "coupling term:",coupling_term
    print "J_x*N/2:",J_x*N/2.0
    print "const:",const
    print "J_x*N:",J_x*N
    print "J_x*N/2:",J_x*N/2.0
    print "sum(hs_x)/2:",sum(hs_x)/2.0
    return field_term + coupling_term + const
    
def from_xs_consts(hs_x,J_x):
    """Given hs, J for xs scheme, return equivalent constants for sigma scheme"""
    hs_sigma = hs_x/2.0 + J_x/2.0
    J_sigma = J_x / 4.0
    N = len(hs_x)
    const = J_x*N/4.0 + np.sum(hs_x)/2.0
    return hs_sigma,J_sigma,const

def from_sigma_consts(hs_sigma,J_sigma):
    J_x = 4*J_sigma
    hs_x = 2*hs_sigma - J_x
    return hs_sigma,J_sigma

# def x_consts(hs_x,J_x):
#     J_sigma = J_x/4.0
#     hs_sigma = (hs_x + J_x)/2.0
#     return hs_
    
def test_conversion(trials):
    N = 10000
    energy_xs = []
    energy_sigmas = []
    for i in xrange(trials):
        hs_x = np.random.random(N)*100 - 50
        J_x = random.random()*100 - 50
        xs = 1*(np.random.random(N) < 0.5)
        sigmas = from_xs(xs)
        hs_sigma, J_sigma, const = from_xs_consts(hs_x,J_x)
        hs_x_check,J_x_check = from_sigma_consts(hs_sigma,J_sigma)
        energy_x = hamil_xs2(xs,hs_x,J_x)
        energy_sigma = hamil_sigmas(sigmas,hs_sigma,J_sigma) + const
        print energy_x,energy_sigma,energy_x == energy_sigma
        print "check:",hs_x -- hs_x_check,J_x == J_x_check
        energy_xs.append(energy_x)
        energy_sigmas.append(energy_sigma)
    plt.scatter(energy_xs,energy_sigmas)
    m = min(energy_xs + energy_sigmas)
    M = max(energy_xs + energy_sigmas)
    plt.plot([m,M],[m,M])
    print pearsonr(energy_xs,energy_sigmas)
    plt.show()

def explore_coupling_const(iterations=1000000):
    """Given 3 state system, explore spin probabilities as function of coupling strength"""
    N = 10
    x0 = [0]*N
    hs = [log(1000000)]*N
    def hamil(xs,J):
        return dot(xs,hs) + J*(xs[0] + sum([xi*xj for (xi,xj) in pairs(xs)]))
    Js = interpolate(-16,-8+1,20)
    def proposal(xs):
        return [int(random.random() < 0.5) for i in range(N)]
    results = []
    for J in Js:
        chain = mh(f=lambda xs:-hamil(xs,J),proposal=proposal,x0=x0,use_log=True,iterations=iterations)
        ps = map(mean,transpose(chain))
        results.append((J,ps))
    Js,pss = transpose(results)
    pss = transpose(pss)
    colors = "bgrcmyk"
    for i,ps in enumerate(pss):
        color = colors[i % len(colors)]
        plt.plot(Js,ps,marker='o',linestyle='',color=color)
        errs = [p+1.96*sqrt(p*(1-p)/iterations)**(i+1) + p**(i+1) for p in pss[0]]
        print i,errs
        plt.plot(Js,[p**(i+1) for p in pss[0]])
        # plt.errorbar(Js,[p**(i+1) for p in pss[0]],yerr=errs,
        #              marker='',linestyle='--',color=color)
    plt.plot(Js,[1.0/iterations for J in Js])
    #plt.semilogy()
    return results

def ising(hs,J,iterations=50000,boundary="periodic",spins=None,burn_in=0):
    N = len(hs)
    if spins is None:
        spins = np.array([random.choice([-1,1]) for i in range(N)])
    occupancies = np.zeros(N)
    for t in verbose_gen(xrange(iterations),modulus=1000):
        for i in range(N):
            current_energy = spins[i]*(hs[i] + J * (spins[(i-1)%N] + spins[(i+1)%N]))
            prop_energy = - current_energy
            p_prop = exp(-prop_energy)/(exp(-current_energy) + exp(-prop_energy))
            #print "p_prop:",p_prop
            if random.random() < p_prop:
                spins[i] *= -1
        if t % 1000 == 0:
            print sum(spins)
        if t > burn_in:
            occupancies += (spins == 1)
        #print "magnetization:",np.sum(spins == 1)
    return occupancies/(iterations-burn_in)

def ising_last(hs,J,iterations,spins=None):
    N = len(hs)
    if spins is None:
        spins = np.array([random.choice([-1,1]) for i in range(N)])
    occupancies = np.zeros(N)
    for t in xrange(iterations):
        for i in range(N):
            current_energy = spins[i]*(hs[i] + J * (spins[(i-1)%N] + spins[(i+1)%N]))
            prop_energy = - current_energy
            p_prop = exp(-prop_energy)/(exp(-current_energy) + exp(-prop_energy))
            #print "p_prop:",p_prop
            if random.random() < p_prop:
                spins[i] *= -1
        occupancies += (spins == 1)
        #print "magnetization:",np.sum(spins == 1)
    return spins

def average_spins(spin_samples):
    cols = transpose(spin_samples)
    return [mean(map(lambda c:(c+1)/2.0,col)) for col in cols]
    
    
def cftp(hs,J):
    N = len(hs)
    seed = random.randint(0,2**32-1)
    iterations = 1
    def comp(sigmas1,sigmas2):
        """partial order on spins.  Return -1 if s1 <= s2, 1 if s1 >= 2, 0 otherwise"""
        stat = sum(s1 <= s2 for s1,s2 in zip(sigmas1,sigmas2))
        if stat == N:
            return -1
        elif stat == -N:
            return 1
        else:
            return 0
    def update_spins(spins,i,r):
        current_energy = spins[i]*(hs[i] + J * (spins[(i-1)%N] + spins[(i+1)%N]))
        prop_energy = - current_energy
        p_prop = exp(-prop_energy)/(exp(-current_energy) + exp(-prop_energy))
                #print "p_prop:",p_prop
        if r < p_prop:
            spins[i] *= -1
    converged = False
    rs = []
    while not converged:
        min_state = [-1] * N
        max_state = [1] * N
        #print "iterations:",iterations,sum(min_state),sum(max_state)
        rs = ([random.random() for _ in xrange(iterations) for __ in xrange(N)]) + rs
        for iter in xrange(iterations):
            for i in range(N):
                r = rs[(iter*N + i)]
                update_spins(min_state,i,r)
                update_spins(max_state,i,r)
            #print "discrepancy:",sum(min_state)+sum(max_state)
        if min_state == max_state:
            converged = True
        else:
            iterations *= 2
    print "coupling time:",iterations
    return min_state
        
def cftp_ising(hs,J,replicas):
    samples = [cftp(hs,J) for i in verbose_gen(xrange(replicas))]
    cols = transpose(samples)
    return [mean(map(lambda c:(c+1)/2.0,col)) for col in cols]

def summary_state(hs,J):
    N = len(hs)
    successes = 0
    def update_spin(state,i,r):
        nbhd = state[(i-1)%N],state[(i+1)%N]
        def proj(ch):
            return {'+':[1],
                    '-':[-1],
                    '?':[1,-1]}[ch]
        ustates = product(*[proj(s) for s in nbhd]) # microstates
        updates = [ustate_update(ustate,i,r) for ustate in ustates]
        if len(set(updates)) == 1:
            #print "success:",i
            return updates[0]
        else:
            return '?'
    def ustate_update(ustate,i,r):
        up_energy = (hs[i] + J * (ustate[0] + ustate[1]))
        down_energy = - up_energy
        p_up = exp(-up_energy)/(exp(-up_energy) + exp(-down_energy))
                #print "p_prop:",p_prop
        if r < p_up:
            #print ustate,p_prop,'+'
            return '+'
        else:
            #print ustate,p_prop,'+'
            return '-'
    converged = False
    iterations = 1
    rs = []
    while not converged:
        print "iterations:",iterations
        state = ["?"]*N
        rs = ([random.random() for _ in xrange(iterations) for __ in xrange(N)]) + rs
        for iter in xrange(iterations):
            for i in xrange(N):
                r = rs[(iter*N + i)]
                prev = state[i]
                state[i] = update_spin(state,i,r)
                if state[i] != prev:
                    print state
        if not '?' in state:
            converged = True
        iterations *= 2
        print state
    return [1 if c == '+' else -1 for c in state]
        
            
        
def multiple_ising(hs,J,iterations=50000,replicas=3,method=ising,burn_in=0):
    occ_list = []
    for i in range(replicas):
        print "replica ",i
        occ_list.append(method(hs,J,iterations)[burn_in:])
    #occ_list = [ising(hs,J,iterations) for i in range(replicas)]
    cols = [[(s+1)/2 for s in col] for col in transpose(occ_list)]
    means = map(mean,cols)
    sds = map(sd,cols)
    cis = [1.96*s for s in sds]
    plt.errorbar(range(len(cols)),means,yerr=cis)
    # for occs in occ_list:
    #     plt.plot(occs)

def rgeom(lamb):
    x = 0
    while random.random() < lamb:
        x += 1
    return x
    
def independence_ising(hs,J,iterations=50000):
    """Sample Ising model using independence sampler"""
    N = len(hs)
    def hamil(spins):
        return sum(h[i]*spins[i] + J*spins[i]*spins[(i+1)%N] for i in range(N))
    def rprop(spins):
        return
    pass
        
def continuum_ising(hs,J,spin_ps=None,tol=10**-2):
    """Try to approximate ising via continuum"""
    N = len(hs)
    if spin_ps is None:
        spin_ps = np.array([0.5]*N)
    else:
        spin_ps = np.array(spin_ps)
    old_spin_ps = np.zeros(N)
    def new_ps(i):
        # P(up) = \sum P(up|config)*P(config)
        pl =  spin_ps[(i-1)%N]
        pr =  spin_ps[(i+1)%N]
        hi = hs[i]
        p_up = sum(prob_from_ep((hi+J*(j+k)))*pj*pk for ((j,pj),(k,pk)) in product([(1,pl),(-1,1-pl)],
                                                                                   [(1,pr),(-1,1-pr)]))
        return p_up
    
    while np.sum(np.abs(spin_ps - old_spin_ps)) > tol:
        old_spin_ps = np.array(spin_ps)
        dpdt = np.array([new_ps(i) for i in range(N)]) - spin_ps
        #print dpdt
        spin_ps += 1 * dpdt
        print np.sum(spin_ps),np.sum(old_spin_ps)
    return spin_ps

def mh_ising(hs,J,iterations=50000,verbose=False):
    sigmas = [random.choice([-1,1]) for i in hs]
    N = len(hs)
    iterations *= N # iterations per spin
    def hamil(ss):
        return sum([s*h for (s,h) in zip(ss,hs)]) + J*(sum(ss[i]*ss[(i+1)%N] for i in range(N)))
    def f(ss):
        return -hamil(ss)
    def prop(ss):
        i = random.randrange(N)
        ss_new = ss[:]
        ss_new[i] *= -1
        return ss_new
    chain = mh(f,prop,sigmas,iterations=iterations,verbose=verbose,use_log=True)
    return map(lambda spin:mean([(s + 1)/2 for s in spin]),transpose(chain))
        
def prob_from_ep(epsilon):
   return exp(-epsilon)/(exp(-epsilon) + exp(epsilon))
   
def compute_mean_field(hs,J):
    z = 2 # coordination number
    beta = 1
    f = lambda m,hi:tanh(beta*(hi + J*z*m))
    magnetizations = [converge2(lambda m:f(m,hi),0) for hi in hs]
    return magnetizations
    


def rfim(hs,J):
    """Given system of N spins with hs, J in xs-format, compute occupancies"""
    N = len(hs)
    x0 = [0]*N
    if hs is None:
        hs = [log(1000000)]*N
    def hamil(xs,J):
        return dot(xs,hs) + J*(sum([xi*xj for (xi,xj) in pairs(xs)]))
    states = list(product(*[[0,1] for i in range(N)]))
    weights = [exp(-hamil(state,J)) for state in states]
    Z = sum(weights)
    # def ith_prob(i):
    #     return sum(w for s,w in zip(states,weights) if s[i] == 1)/Z
    out = [0] * N
    for w,state in zip(weights,states):
        for i in xrange(N):
            out[i] += state[i]*w
    return [o/Z for o in out]

def rfim_transmat(hs,J):
    """compute rfim, but with transfer matrices"""
    N = len(hs)
    Ws = [np.matrix([[exp(-(hi + J)),1],
                     [exp(-hi),      1]]) for hi in hs]
    dWs = [np.matrix([[-exp(-(hi + J)),0],
                      [-exp(-hi)      ,0]]) for hi in hs]
    v0 = np.array([0,1])
    vf = np.array([1,1])
    Z = v0.dot(reduce(lambda wi,wj:wi.dot(wj),Ws)).dot(vf)[0,0]
    def pi(i):
        Vs = Ws[0:i] + [dWs[i]] + Ws[i+1:]
        #print len(Ws[0:i]), len([dWs[i]]),len(Ws[i+1:]),(len(Ws[0:i])+len([dWs[i]])+len(Ws[i+1:]))
        #print (Vs)
        numer = (-v0.dot((reduce(lambda wi,wj:wi.dot(wj),Vs))).dot(vf)[0,0])
        return numer/Z
    return [pi(i) for i in range(N)]

def rfim2(ps,lamb):
    hs, J = match_parameters2(ps,lamb)
    # Ws = [np.matrix([[(1-lamb),lamb],
    #                  [p       ,1-p]]) for p in ps]
    Ws = [np.matrix([[exp(-(hi + J)),1],
                     [exp(-hi),      1]]) for hi in hs]
    dWs = [np.matrix([[-exp(-(hi + J)),0],
                      [-exp(-hi)      ,0]]) for hi in hs]
    v0 = np.array([0,1])
    vf = np.array([1,1])
    Z = v0.dot(reduce(lambda wi,wj:wi.dot(wj),Ws)).dot(vf)[0,0]
    def pi(i):
        Vs = Ws[0:i] + [dWs[i]] + Ws[i+1:]
        #print len(Ws[0:i]), len([dWs[i]]),len(Ws[i+1:]),(len(Ws[0:i])+len([dWs[i]])+len(Ws[i+1:]))
        #print (Vs)
        numer = (-v0.dot((reduce(lambda wi,wj:wi.dot(wj),Vs))).dot(vf)[0,0])
        return numer/Z
    return [pi(i) for i in range(len(ps))]

def rfim3(ps,lamb):
    hs, J = match_parameters3(ps,lamb)
    return rfim_transmat(hs,J)
    
def logit(p):
    return log(p/(1-p))

def epsigma_from_prob(p):
    # solve p = exp(-ep)/(exp(-ep)+exp(ep)) for ep
    return -1/2.0*logit(p)

def epx_from_prob(p):
    return -logit(p)
    
def match_parameters(ps,mean_frag_length):
    """Given ps, lambda, find appropriate parameters in xs-based ising model"""
    hxs = [epx_from_prob(p) for p in ps]
    #Jsigma = -atanh(exp(-1.0/mean_frag_length))
    #Jsigma = -(exp(-1.0/mean_frag_length))
    #Jx = 4 * Jsigma
    Jx =  Jsigma
    print mean(hxs),Jx
    return hxs,Jx

def match_parameters2(ps,lamb):
    hs = [-log(p) for p in ps]
    J = log(p/(1-lamb))
    return hs,J

def match_parameters3(ps,lamb):
    hs = [log((1-p)/p) for p in ps]
    J = log(lamb/(1-lamb))
    return hs,J

def plot_mfl_vs_correlation(mfls,N=100,ps=None):
    """
    plot coupling strength as function of mfl
    """
    if ps is None:
        ps = [1.0/10**6 if i != 0 else 0.9999 for i in range(N)]
    hs = [epx_from_prob(p) for p in ps]
    Js = []
    for mfl in mfls:
        f = lambda j:rfim_transmat(hs,j)[mfl] - exp(-1)
        j = bisect_interval(f,-25,0,tolerance=1e-2,verbose=True)
        print mfl,j
        Js.append(j)
    plt.plot(mfls,Js)
    plt.xlabel("MFL")
    plt.ylabel("J")
    return mfls,Js

def main():
    hs = [0]*1000
    hs[500] = -2
    J = -2
    plt.plot(ising(hs,J))
    plt.plot(continuum_ising(hs,J))

