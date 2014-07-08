import random
from utils import pairs

def valid(xs,sigma):
    """
    Determine whether configuration is valid
    """
    ys = sorted(xs)
    return (all(xp - x >= 2*sigma for (x,xp) in pairs(ys)) and
            ys[0] > sigma and ys[-1] < 1 - sigma)

def direct_sample(K,density,return_efficiency=False):
    """
    Sample configuration of K clothespins of length sigma on unit
    interval.
    """
    sigma = density/(2.0*K)
    iterations = 1
    while True:
        xs = [random.random() for i in range(K)]
        if valid(xs,sigma):
            if return_efficiency:
                return 1/float(iterations)
            return xs
        iterations += 1

def rsa(K,density):
    sigma = density/(2.0*K)
    xs = []
    while len(xs) < K:
        x = random.random()
        if valid(xs + [x],sigma):
            xs.append(x)
    return xs

def accessible_regions(xs,sigma):
    """Given a valid configuration xs, return a list of tuples
    denoting accessible regions where another disc can be placed"""
    borders = [0,1]
    for x in xs:
        borders.extend([x-sigma,x+sigma])
    interiors = map(tuple,group_by(sorted(borders),2))
    return [(a+sigma,b-sigma) for (a,b) in interiors if (b-sigma) - (a+sigma) > 0]
    
    
def smart_rsa(K,density):
    sigma = density/(2.0*K)
    xs = []
    while True:
        while len(xs) < K:
            acc_regs = accessible_regions(xs,sigma)
            if acc_regs == []:
                #print "no accessible regions"
                xs = []
                continue
            region_indices = range(len(acc_regs))
            region_probs = normalize([(b-a) for (a,b) in acc_regs])
            reg_j = inverse_cdf_sample(region_indices,region_probs)
            (a,b) = acc_regs[reg_j]
            r = random.random()
            x = (b-a)*r + a
            assert valid([x] + xs,sigma),(x,xs)
            xs.append(x)
        return xs
