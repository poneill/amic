"""
This file implements the Gillespie algorithm for simple chromosomal systems (G,q); no overlap exclusion or interactions
"""
import random
from utils import inverse_cdf_sample,normalize,zipWith,mean
from math import exp
import numpy as np

beta = 1
G = 5000000
q = 10
eps = [random.gauss(0,1) for i in range(G)]
#eps = [0 for i in range(G)]

koffs = [exp(-beta*ep) for ep in eps] # koffs to emphasize that the
                                      # off-rates differ; on-rates are
                                      # scaled to unity.

def update(chromosome,qf,koffs,verbose=False):
    """Do a single iteration of SSA

    Given:
    chromosome: binary vector describing occupation state of ith site
    qf: number of free copies in cytosol
    koffs: vector of off-rates

    Return:
    updated_chromosome
    updated_qf
    time at which transition chromosome -> updated_chromosome occurs"""
    # Determine which reactions can occur
    # all free sites can become bound at rate qf * 1
    # all bound sites can become unbound at rate koff_i
    rates = [koffs[i] if bs else qf for i,bs in enumerate(chromosome)]
    sum_rate = sum(rates)
    idx = inverse_cdf_sample(range(G),normalize(rates))
    time = random.expovariate(sum_rate)
    updated_chromosome = chromosome[:]
    if chromosome[idx]: # if reaction is an unbinding reaction...
        if verbose:
            print "unbinding at: ",idx
        updated_chromosome[idx] = 0
        updated_qf = qf + 1
    else: # a binding reaction...
        if verbose:
            print "binding at: ",idx
        updated_chromosome[idx] = 1
        updated_qf = qf - 1
    return updated_chromosome,updated_qf,time

def sample_path_ref(qf,koffs,t_final,chromosome=None,verbose=False):
    """Simulate a sample path until time t_final and return the marginal occupancies"""
    if chromosome is None: # then start from empty chromosome
        chromosome = [0] * G
    chrom = chromosome[:]
    new_chrom = chrom[:]
    occupancies = [0 for c in chromosome]
    t = 0
    dt = 0
    while t < t_final:
        new_chrom,qf,dt = update(chrom,qf,koffs,verbose=verbose)
        t += dt
        # This ugly bit of code ensures that we only track the
        # occupancies until exactly time t_final.
        if t > t_final:
            dt = t_final - t + dt
        occupancies = zipWith(lambda occ,ch:occ + ch*dt,occupancies,chrom)
        chrom = new_chrom[:]
        if verbose:
            print "t:",t,"dt:",dt,"q:",qf,"qbound:",sum(chrom),"mean occ:",sum([occ/t for occ in occupancies])
    return [occ/t_final for occ in occupancies]
        
def sample_path_ref2(qf,koffs,t_final,chromosome=None,verbose=False):
    """Simulate a sample path until time t_final and return the marginal occupancies.
    Integrates update, sample_path_ref framework.
    """
    if chromosome is None: # then start from empty chromosome
        chromosome = [0] * G
    t = 0
    dt = 0
    occs = [0 for c in chromosome]
    while t < t_final:
        rates = [koffs[i] if bs else qf for i,bs in enumerate(chromosome)]
        sum_rate = sum(rates)
        dt = random.expovariate(sum_rate)
        t += dt
        if t > t_final:
            dt = t_final - t + dt
        # update occupancies after deciding dt, before updating chromosome
        occs = zipWith(lambda occ,ch:occ + ch*dt,occs,chromosome)
        idx = inverse_cdf_sample(range(G),normalize(rates))
        if chromosome[idx]: # if reaction is an unbinding reaction...
            if verbose:
                print "unbinding at: ",idx
            chromosome[idx] = 0
            qf += 1
        else: # a binding reaction...
            if verbose:
                print "binding at: ",idx
            chromosome[idx] = 1
            qf -= 1
        if verbose:
            print "t:",t,"dt:",dt,"q:",qf,"qbound:",sum(chromosome),"mean occ:",sum([occ/t for occ in occs])
    return [occ/t_final for occ in occs]

def sample_path(qf,koffs,t_final,chromosome=None,verbose=False):
    """Simulate a sample path until time t_final and return the marginal occupancies.
    Integrates update, sample_path_ref framework.
    """
    if chromosome is None: # then start from empty chromosome
        chromosome = [0] * G
    t = 0
    dt = 0
    occs = [0 for c in chromosome]
    while t < t_final:
        rates = [koffs[i] if bs else qf for i,bs in enumerate(chromosome)]
        sum_rate = sum(rates)
        dt = random.expovariate(sum_rate)
        t += dt
        if t > t_final:
            dt = t_final - t + dt
        # update occupancies after deciding dt, before updating chromosome
        #occs = zipWith(lambda occ,ch:occ + ch*dt,occs,chromosome) # substantial speedup by iterating...
        for i in xrange(G):
            occs[i] += chromosome[i]*dt
        idx = inverse_cdf_sample(range(G),normalize(rates))
        if chromosome[idx]: # if reaction is an unbinding reaction...
            if verbose:
                print "unbinding at: ",idx
            chromosome[idx] = 0
            qf += 1
        else: # a binding reaction...
            if verbose:
                print "binding at: ",idx
            chromosome[idx] = 1
            qf -= 1
        if verbose:
            print "t:",t,"dt:",dt,"q:",qf
    return [occ/t_final for occ in occs]

def sample_path_np(qf,koffs,t_final,chromosome=None,verbose=False):
    """Simulate a sample path until time t_final and return the marginal occupancies.
    Integrates update, sample_path_ref framework.
    """
    G = len(koffs)
    if chromosome is None: # then start from empty chromosome
        chromosome = np.zeros(G)
    t = 0
    dt = 0
    occs = np.zeros(G)
    while t < t_final:
        rates = [koffs[i] if bs else qf for i,bs in enumerate(chromosome)]
        sum_rate = sum(rates)
        dt = random.expovariate(sum_rate)
        t += dt
        if t > t_final:
            dt = t_final - t + dt
        # update occupancies after deciding dt, before updating chromosome
        #occs = zipWith(lambda occ,ch:occ + ch*dt,occs,chromosome) # substantial speedup by iterating...
        # for i in xrange(G):
        #     occs[i] += chromosome[i]*dt
        occs += chromosome*dt
        idx = inverse_cdf_sample_unnormed(range(G),rates)
        if chromosome[idx]: # if reaction is an unbinding reaction...
            if verbose:
                print "unbinding at: ",idx
            chromosome[idx] = 0
            qf += 1
        else: # a binding reaction...
            if verbose:
                print "binding at: ",idx
            chromosome[idx] = 1
            qf -= 1
        if verbose:
            print "t:",t,"dt:",dt,"q:",qf
    return occs/t_final

def inverse_cdf_sample_unnormed_ref(xs,ws):
    r = random.random() * sum(ws)
    acc = 0
    for x,w in zip(xs,ws):
        acc += w
        if acc > r:
            return x

def inverse_cdf_sample_unnormed(xs,ws):
    r = random.random() * sum(ws)
    acc = 0
    for i,w in enumerate(ws):
    #for x,w in zip(xs,ws):
        acc += w
        if acc > r:
            return xs[i]
