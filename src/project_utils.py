import random
from math import exp,factorial,sqrt

def random_site(n):
    """Generate a random sequence of n nucleotides"""
    return "".join(random.choice("ACGT") for i in range(n))

def inverse_cdf_sample(xs,ps):
    """Sample from discrete distribution ps over set xs"""
    r = random.random()
    acc = 0
    for x,p in zip(xs,ps):
        acc += p
        if acc > r:
            return x

def pprint(xs):
    """Pretty print a nested list"""
    for x in xs:
        print x
    
def rpois(_lambda):
    """Sample Poisson rv"""
    L = exp(-_lambda)
    k = 0
    p = 1
    while p > L:
        k += 1
        u = random.random()
        p *= u
    return k - 1

def dpois(k,_lambda):
    """pmf of Poisson rv"""
    return exp(-_lambda)*_lambda**k/factorial(k)

def mean(xs):
    return sum(xs)/float(len(xs))

def variance(xs,correct=True):
    n = len(xs)
    correction = n/float(n-1) if correct else 1
    mu = mean(xs)
    return correction * mean([(x-mu)**2 for x in xs])

def sd(xs,correct=True):
    return sqrt(variance(xs,correct=correct))

def pearsonr(xs,ys):
    mu_x = mean(xs)
    mu_y = mean(ys)
    return (mean([(x-mu_x)*(y-mu_y) for (x,y) in zip(xs,ys)])/
            (sd(xs,correct=False)*sd(ys,correct=False)))

def normalize(xs):
    total = float(sum(xs))
    return [x/total for x in xs]
