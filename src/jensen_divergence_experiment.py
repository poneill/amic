"""

Theory predicts that the acceptance rate for MH independence
sampling of P from proposal Q should be exp(-2*DJS(P||Q)).  Is this true?

"""

from utils import normalize,inverse_cdf_sample,verbose_gen,dot,mean
from math import log,exp,sqrt
import random
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

# if not 'P' in dir():
#     print "defining P and Q"
#     P = normalize([random.random() for i in range(K)])
#     Q = normalize([random.random() for i in range(K)])

def sample_from(P):
    return inverse_cdf_sample(range(len(P)),P)

def rcat(K):
    return normalize([random.random() for i in range(K)])

def Dkl(P,Q):
    K = len(P)
    return sum([P[i]*log(P[i]/Q[i]) for i in range(K)])

def Djs(P,Q):
    return (Dkl(P,Q) + Dkl(Q,P))/2.0

def pred_acceptance(P,Q):
    return exp(-2*Djs(P,Q))

def pred_acceptance2(P,Q):
    K = len(P)
    return sum(P[i]*Q[j] for i in range(K) for j in range(K))

def pred_acceptance3(P,Q):
    """This is correct"""
    K = len(P)
    return sum(min(P[x]*Q[xp],P[xp]*Q[x]) for x in range(K) for xp in range(K))

def test_triangle_inequality(K=5,trials=10000):
    xs = []
    ys = []
    for trial in xrange(trials):
        ps = simplex_sample(K)
        qs = simplex_sample(K)
        rs = simplex_sample(K)
        ar_pq = pred_acceptance3(ps,qs)
        ar_qr = pred_acceptance3(qs,rs)
        ar_pr = pred_acceptance3(ps,rs)
        print ar_pr < ar_pq + ar_qr
        xs.append(ar_pr)
        ys.append(ar_pq + ar_qr)
    plt.scatter(xs,ys)

def test_discrete_metric(K=5,trials=10000):
    """
    Nice, clean inverse relationship: distance gives lower bound on
    acceptance ratio
    """
    distances = []
    ars = []
    for trial in xrange(trials):
        ps = simplex_sample(K)
        qs = simplex_sample(K)
        distances.append(discrete_metric(ps,qs)/K)
        ars.append(pred_acceptance3(ps,qs))
    plt.scatter(distances,ars,marker='.')
    plt.plot(*pl(lambda x:1-x,interpolate(0.001,1,1000)))
    plt.show()
        
def mh_sample(P,Q,iterations=10000):
    xs = []
    x = sample_from(Q)
    for i in xrange(iterations):
        xp = sample_from(Q)
        a = min(P[xp]/P[x]*Q[x]/Q[xp],1)
        r = random.random()
        if r < a:
            x = xp
        xs.append(x)
    return xs

def mh_acceptance(P,Q,iterations=10000):
    x = sample_from(Q)
    acceptances = 0
    acceptance_ratios = []
    for i in xrange(iterations):
        xp = sample_from(Q)
        a = min(P[xp]/P[x]*Q[x]/Q[xp],1)
        acceptance_ratios.append(a)
        r = random.random()
        if r < a:
            x = xp
            acceptances += 1
    # print "mean acceptances:",acceptances/float(iterations)
    # print "mean acceptance ratio:",mean(acceptance_ratios)
    #return mean(acceptance_ratios)
    return acceptances/float(iterations)
        
def systematic_exp(K=10,trials=100,mh_trials=10000,plotting=False):
    norms = []
    normKs = []
    theory = []
    pred3s = []
    obs = []
    for trial in verbose_gen(xrange(trials)):
        P = rcat(K)
        Q = rcat(K)
        a = mh_acceptance(P,Q,mh_trials)
        norm = l2(P,Q)
        normK = norm*K
        pred3 = pred_acceptance3(P,Q)
        norms.append(norm)
        normKs.append(normK)
        obs.append(a)
        pred3s.append(pred3)
    fcorrect = poly1d(polyfit(normKs,obs,1))
    if plotting:
        plt.scatter(norms,obs)
        plt.scatter(map(fcorrect,normKs),obs,color='green')
        plt.scatter(pred3s,obs,color='blue')
        plt.plot([0,1],[0,1])
    print "fit to norm:",polyfit(norms,obs,1)
    print pearsonr(norms,obs)
    print "fit to normK:",polyfit(normKs,obs,1)
    print pearsonr(normKs,obs)
    print "fit to Pred3:",polyfit(pred3s,obs,1)
    print pearsonr(pred3s,obs)
    return theory,obs

def find_relationship(trials=100,mh_trials=10000):
    Ks = range(2,200,10)
    ms = []
    bs = []
    for K in Ks:
        theory,obs = systematic_exp(K,trials=trials,mh_trials=mh_trials)
        m,b = polyfit(theory,obs,1)
        print "K:",K,"m:",m,"b:",b
        ms.append(m)
        bs.append(b)
    plt.scatter(Ks,ms)
    return Ks,ms
    
def softmin(a,b,beta):
    return -log(exp(-beta*a) + exp(-beta*b))/beta

def pred3_vs_djs(K=100):
    trials = 100
    pred3s = []
    dklpqs = []
    dklqps = []
    pred_djss = []
    for trial in xrange(trials):
        P,Q = rcat(K),rcat(K)
        pred3s.append(pred_acceptance3(P,Q))
        pred_djss.append(pred_acceptance(P,Q))
        dklpqs.append(Dkl(P,Q))
        dklqps.append(Dkl(Q,P))
    plt.scatter(pred3s,pred_djss)
    print "JS:",pearsonr(pred3s,pred_djss)
    plt.scatter(pred3s,dklpqs,color='green')
    print "Dkl(P||Q):",pearsonr(pred3s,dklpqs)
    plt.scatter(pred3s,dklqps,color='red')
    print "Dkl(Q||P):",pearsonr(pred3s,dklqps)
    plt.plot([0,1],[0,1])
    plt.show()
        
def two_step_mh(P,Q,R,iterations=100000):
    xs = []
    xp = sample_from(R)
    x = sample_from(R)
    accs1 = 0
    accs2 = 0
    for it in verbose_gen(xrange(iterations),10000):
        xpp = sample_from(R)
        a1 = min(Q[xpp]/Q[xp]*R[xp]/R[xpp],1)
        if random.random() < a1:
            xp = xpp
            accs1 += 1
        a2 = min(P[xp]/P[x]*Q[x]/Q[xp],1)
        if random.random() < a2:
            x = xp
            accs2 += 1
        xs.append(x)
    print "first stage acceptance:",accs1/float(iterations)
    print "second stage acceptance:",accs2/float(iterations)
    return xs

def two_step_shuffle(P,Q,R,iterations=100000):
    xs = []
    xps = mh_sample(Q,R,iterations=iterations)
    random.shuffle(xps)
    x = xps[0]
    #qs = [sample_from(Q) for i in xrange(iterations)]
    for i in xrange(iterations):
        xp = xps[i]
        #xp = qs[i]
        a2 = min(P[xp]/P[x]*Q[x]/Q[xp],1)
        if random.random() < a2:
            x = xp
        xs.append(x)
    return xs

def freqs(xs,K):
    N = float(len(xs))
    return [xs.count(i)/N for i in range(K)]
        
