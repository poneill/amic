"""
This file implements the Gillespie algorithm for simple chromosomal systems (G,q); no overlap exclusion or interactions
"""
import random
from utils import inverse_cdf_sample,normalize,zipWith,mean
from math import exp
import numpy as np
from project_utils import inverse_cdf_sampler

beta = 1
G = 5000000
q = 10
if not "eps" in dir():
    eps = [random.gauss(0,1) for i in range(G)]
    #eps = [0 for i in range(G)]
    koffs = [exp(beta*ep) for ep in eps] # koffs to emphasize that the
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
        chromosome = [0] * len(koffs)
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
        chromosome = [0] * len(koffs)
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

def update_spatial(chromosome,qf,koffs,verbose=False):
    """Do a single iteration of SSA with sliding reactions included.

    Given:
    
    chromosome: binary vector describing occupation state of ith site.
    Each site can be unbound (0), bound non-specifically(1) or bound
    specifically(2).  Non-specific binding has a free energy of -7
    kbt, sequence-specific binding is as usual.  A TF bound
    non-specifically can slide 1 bp left or right with no change in
    free energy.
    
    qf: number of free copies in cytosol
    koffs: vector of off-rates

    Return:
    updated_chromosome
    updated_qf
    time at which transition chromosome -> updated_chromosome occurs

    """
    # Determine which reactions can occur
    # free copies can bind non-specifically at rate ep_ns
    # non-specifically bound copies can transition to specific binding or slide left or right
    # specifically bound copies can transition to non-specific binding
    ep_ns = -7
    k_ns = exp(-beta*ep_ns)
    ep_slide = 0
    k_slide = exp(-beta*ep_slide) # 1 obviously
    k1 = 1 # rate for reactions that happen on default simulation timescale
    G = len(chromosome)
    reactions = [(i,'N',qf*k_ns) for i in xrange(G)]
    for i,c in enumerate(chromosome):
        # if c == 0 and qf > 0:
        #     reactions.append((i,'N',qf*k_ns))
        if c == 1: # if bound non-specifically
            # tf can bind specifically
            reactions.append((i,'S',k1))
            # tf can fall off 
            reactions.append((i,'F',k1))
            # tf can slide
            if chromosome[(i-1)%G] == 0:
                reactions.append((i,'L',k1))
            if chromosome[(i+1)%G] == 0:
                reactions.append((i,'R',k1))
        elif c == 2:
            reactions.append((i,'U',koffs[i]))
    ###
    rates = [reaction[2] for reaction in reactions]
    sum_rate = sum(rates)
    chr_idx,rx_type,rate = inverse_cdf_sample(reactions,normalize(rates))
    time = random.expovariate(sum_rate)
    if verbose:
        print chr_idx,rx_type,rate,time
    updated_chromosome = chromosome[:]
    if rx_type == 'N': # tf binds non-specifically
        updated_chromosome[chr_idx] = 1
        updated_qf = qf - 1
    elif rx_type == 'S': # tf transitions to specific binding
        updated_chromosome[chr_idx] = 2
        updated_qf = qf
    elif rx_type == 'F':
        updated_chromosome[chr_idx] = 0
        updated_qf = qf + 1
    elif rx_type == 'L':
        updated_chromosome[chr_idx] = 0
        updated_chromosome[(chr_idx-1)%G] = 1
        updated_qf = qf
    elif rx_type == 'R':
        updated_chromosome[chr_idx] = 0
        updated_chromosome[(chr_idx+1)%G] = 1
        updated_qf = qf
    elif rx_type == 'U':
        updated_chromosome[chr_idx] = 1
        updated_qf = qf
    else:
        print "Didn't recognize reaction type:",rx_type
        assert False
    return updated_chromosome,updated_qf,time

def update_spatial_xs(q,ns_bound,s_bound,koffs,verbose=False):
    """
    Given:

    q: total copy number
    ns_bound: list of ns_bounde copies
    s_bound: list of specifically bound copies
    koffs: chromosomal off-rates

    return:
    updated xs
    """
    ep_ns = -7
    k_ns = exp(-beta*ep_ns)
    ep_slide = 0
    k_slide = exp(-beta*ep_slide) # 1 obviously
    k1 = 1 # rate for reactions that happen on default simulation timescale
    G = len(koffs)
    qf = q - (len(ns_bound) + len(s_bound))
    print "qf:",qf
    reactions = [(i,'N',qf*k_ns) for i in xrange(G)]
    for i in ns_bound:
        # tf can bind specifically
        reactions.append((i,'S',k1))
        # tf can fall off 
        reactions.append((i,'F',k1))
        # tf can slide
        if not (i-1)% G in ns_bound + s_bound:
            reactions.append((i,'L',k1))
        if not (i+1)% G in ns_bound + s_bound:
            reactions.append((i,'R',k1))
    for i in s_bound:
        reactions.append((i,'U',koffs[i]))
    ###
    rates = [reaction[2] for reaction in reactions]
    sum_rate = sum(rates)
    chr_idx,rx_type,rate = inverse_cdf_sampler(reactions,normalize(rates))()
    time = random.expovariate(sum_rate)
    updated_ns_bound = ns_bound[:]
    updated_s_bound = s_bound[:]
    if verbose:
        print chr_idx,rx_type,rate,time,"ns:",len(ns_bound),"s:",len(s_bound)
    if rx_type == 'N': # tf binds non-specifically
        updated_ns_bound.append(chr_idx)
    elif rx_type == 'S': # tf transitions to specific binding
        updated_ns_bound.remove(chr_idx)
        updated_s_bound.append(chr_idx)
    elif rx_type == 'F':
        updated_ns_bound.remove(chr_idx)
    elif rx_type == 'L':
        updated_ns_bound.remove(chr_idx)
        updated_ns_bound.append((chr_idx-1)%G)
    elif rx_type == 'R':
        updated_ns_bound.remove(chr_idx)
        updated_ns_bound.append((chr_idx+1)%G)
    elif rx_type == 'U':
        updated_s_bound.remove(chr_idx)
        updated_ns_bound.append(chr_idx)
    else:
        print "Didn't recognize reaction type:",rx_type
        assert False
    return updated_ns_bound,updated_s_bound,time

def update_spatial_dict(q,bound,koffs,verbose=False):
    ep_ns = -7
    k_ns = exp(-beta*ep_ns)
    ep_slide = 0
    k_slide = exp(-beta*ep_slide) # 1 obviously
    k1 = 1 # rate for reactions that happen on default simulation timescale
    G = len(koffs)
    qf = q - len(bound)
    reactions = [(None,"N",qf*k_ns*(G-len(bound)))] # wrap up all non-specific binding reactions into one
    for i,status in bound:
        if status == 'N': #non-specific
            # tf can bind specificially
            reactions.append((i,'S',k1))
            # tf can fall off 
            reactions.append((i,'F',k1))
            # tf can slide
            if not (i-1)% G in bound:
                reactions.append((i,'L',k1))
            if not (i+1)% G in bound:
                reactions.append((i,'R',k1))
        else: #bound specifically
            reactions.append((i,'U',koffs[i]))
        
def sample_path_spatial_xs(q,koffs,t_final,verbose=False):
    occupancies = [0 for c in koffs]
    s_bound = []
    ns_bound = []
    t = 0
    while t < t_final:
        ns_bound,s_bound,dt = update_spatial_xs(q,ns_bound,s_bound,koffs,verbose=verbose)
        if t + dt > t_final:
            dt = t_final - t + dt
        for i in ns_bound + s_bound:
            occupancies[i] += dt
        t += dt
    return occupancies
            
def sample_path_spatial_ref(qf,koffs,t_final,chromosome=None,verbose=False):
    """
    Simulate a sample path for the spatial model until time t_final and
    return the marginal occupancies.  Occupancies include both specific and non-specific binding.
    """
    if chromosome is None: # then start from empty chromosome
        G = len(koffs)
        chromosome = [0] * G
    chrom = chromosome[:]
    new_chrom = chrom[:]
    occupancies = [0 for c in chromosome]
    t = 0
    dt = 0
    while t < t_final:
        new_chrom,qf,dt = update_spatial(chrom,qf,koffs,verbose=verbose)
        t += dt
        # This ugly bit of code ensures that we only track the
        # occupancies until exactly time t_final.
        if t > t_final:
            dt = t_final - t + dt
        #occupancies = zipWith(lambda occ,ch:occ + (ch>0)*dt,occupancies,chrom)
        for i in range(G):
            occupancies[i] += int(chrom[i])*dt
        chrom = new_chrom[:]
        if verbose:
            print "t:",t,"dt:",dt,"q:",qf,"non-spec bound:",sum(c == 1 for c in chrom),"spec bound:",sum(c == 2 for c in chrom),"mean occ:",sum([occ/t for occ in occupancies])
    return [occ/t_final for occ in occupancies]

def sample_path_spatial_ref2(qf,koffs,t_final,chromosome=None,verbose=False):
    """
    Simulate a sample path for the spatial model until time t_final and
    return the marginal occupancies.  Occupancies include both specific and non-specific binding.
    """
    if chromosome is None: # then start from empty chromosome
        G = len(koffs)
        chromosome = [0] * G
    chrom = chromosome[:]
    new_chrom = chrom[:]
    occupancies = [0 for c in chromosome]
    t = 0
    dt = 0
    while t < t_final:
        chrom,qf,dt = update_spatial(chrom,qf,koffs,verbose=verbose)
        t += dt
        # This ugly bit of code ensures that we only track the
        # occupancies until exactly time t_final.
        if t > t_final:
            dt = t_final - t + dt
        #occupancies = zipWith(lambda occ,ch:occ + (ch>0)*dt,occupancies,chrom)
        if verbose:
            print "t:",t
        elif random.random() < 0.001:
            print "time:",t
        # This is a big time sink
        # for i in range(G):
        #     occupancies[i] += int(chrom[i] > 0)*dt
        ###
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

def convolve(ps,sigma):
    G = len(ps)
    new_ps = [0]*G
    G = len(ps)
    for i in xrange(G):
        p = ps[i]
        new_ps[i] += p*(1-2*sigma)
        new_ps[(i-1)%G] += p*sigma
        new_ps[(i+1)%G] += p*sigma
    return new_ps
