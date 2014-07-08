"""
Implement Metropolis-Hastings sampler
"""
from math import log
import random
def mh(p,proposal,x0,iterations=50000,use_log=False,verbose=0,capture_state=lambda x:x):
    """General purpose Metropolis-Hastings sampler.

    Inputs:
    p: probability density function, up to a constant

    proposal: function for sampling px_new given px

    iterations: number of rounds of sampling to perform

    use_log: if True, assume that p is the log-transformed density
    function log(p(x)).

    verbose: Set to positive integer to print chain statistics when
    iteration % verbose == 0.

    Outputs:

    xs: the chain of tuples of the form (x,p(x)).
    """
    x = x0
    px = p(x)
    xs = [(capture_state(x),px)]
    px_string = "p(x)" if not use_log else "log(p(x))"
    acceptances = 0
    for i in xrange(1,iterations+1):
        if verbose and i % verbose == 0:
            print "iteration:",i,px_string,px,"acceptance efficiency:",acceptances/float(i)
        x_new = proposal(x)
        px_new = p(x_new)
        acceptance_ratio = px_new/px if not use_log else (px_new - px)
        r = random.random() if not use_log else log(random.random())
        if r < acceptance_ratio:
            x = x_new
            px = px_new
            acceptances += 1
        xs.append((capture_state(x),px))
    if verbose:
        print "Acceptance efficiency:",acceptances/float(iterations)
    return xs
