from math import exp,log,pi
from utils import fac
from scipy import special

def log_choose(N,k):
    return log_fac(N) - (log_fac(k)+log_fac(N-k))

def log_fac(n):
    """compute log(n!)"""
    return special.gammaln(n+1)

def dbinom_ref(N,k,p):
    return choose(N,k)*p**k*(1-p)**(N-k)
