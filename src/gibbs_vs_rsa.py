from sample import *
import random
random.seed("gibbs vs rsa")

def main(filename=None):
    G = 100
    ks = ([exp(-random.gauss(0,10)) for i in xrange(G)])
    max_k = max(ks)
    q = 10
    
    rsa_occs = sample_average([ss_from_xs(gibbs_fast_harness(ks,q,1),G)
                               for i in verbose_gen(range(1000))])
    gibbs_occs = sample_average([ss_from_xs(gibbs_fast_harness(ks,q,10),G)
                                 for i in verbose_gen(range(1000))])
    plt.plot(rsa_occs,label="RSA")
    plt.plot(gibbs_occs,label="Gibbs sampling @ t=10")
    plt.ylabel("Mean Occupancy")
    plt.xlabel("Genomic Coordinate")
    plt.legend(loc=0)
    plt.title("RSA vs. Gibbs Sampling for 100 LN(0,10) sites")
    maybesave(filename)
    
