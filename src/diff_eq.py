"""
In this file, model the system as a system of linear equations
"""
from utils import zipWith

def equilibrate(qtot,koffs,dt=0.01,iterations=1000):
    q = qtot
    ss = [0 for k in koffs]
    def dqdt():
        ans = -q*sum((1-s) for s in ss) + sum(k*s for s,k in zip(ss,koffs))
        #print "dqdt:",ans
        return ans
    def dsidt(i):
        si = ss[i]
        ki = koffs[i]
        ans = q*(1-si) - ki*si
        #print "dsdt ",i,":",ans
        return ans
    for iteration in xrange(iterations):
        dq = dqdt()
        dssdt = [dsidt(i) for i,k in enumerate(koffs)]
        q = q + dq*dt
        ss = zipWith(lambda s,ds:s+ds*dt,ss,dssdt)
        if iteration % 1000 == 0:
            print q#,ss,q + sum(ss)
    return ss
