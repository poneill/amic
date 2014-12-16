"""Utility functions for energy matrices"""

import random
import numpy as np
from project_utils import random_site
from utils import wc

# The 0th row is A, the 1st row is C...
base_index = dict(zip("ACGT",range(4)))

def random_energy_matrix(width,sigma=1):
    """Return a energy matrix of given width containing mean-zero gaussian
    entries with sd sigma"""
    return [[random.gauss(0,sigma) for base in "ACGT"] for i in range(width)]

def null_energy_matrix(width):
    return [[0,0,0,0] for i in range(width)]

def score(energy_matrix,sequence):
    """Compute the score of sequence with energy matrix"""
    return sum(column[base_index[base]] for (base, column) in zip(sequence,energy_matrix))

def rev_comp_matrix(matrix):
    return [row[::-1] for row in matrix[::-1]]
    
def score_genome_np(energy_matrix,genome,both_strands=True):
    """Score entire genome, returning a list of delta G's indexed by left
    endpoint.  If genome is of length L and matrix of width w, binding
    landscape is of length L-w+1"""
    L = len(genome)
    w = len(energy_matrix)
    ext_genome = genome + genome[:w]
    fwd_scores = np.array([score(energy_matrix,ext_genome[i:i+w]) for i in xrange(L)])
    if both_strands:
        rev_matrix = rev_comp_matrix(energy_matrix)
        rev_scores = np.array([score(rev_matrix,ext_genome[i:i+w]) for i in xrange(L)])
        return fwd_scores,rev_scores
    else:
        return fwd_scores

def update_scores_np_ref(scores,i,j,dw,genome,w):
    """Suppose scores = score_genome(energy_matrix,genome), and suppose
    the energy_matrix[i][j]+=dw.  Compute the revised scores."""
    G = len(genome)
    new_scores = np.copy(scores)
    relevant_base = {v:k for (k,v) in base_index.items()}[j]
    for idx in xrange(G):
        if genome[(idx + i) % G] == relevant_base:
            # print "base at idx + i: %s is %s" % (idx + i,genome[idx + i])
            # print "altering score at idx: %s from %s to %s" % (idx,new_scores[idx],new_scores[idx]+dw)
            new_scores[idx] += dw
    return new_scores

def update_scores_np(fwd_scores,rev_scores,fwd_i,fwd_j,dw,w,genome):
    G = len(genome)
    rel_fwd_base = {v:k for (k,v) in base_index.items()}[fwd_j]
    rel_rev_base = wc(rel_fwd_base)
    rev_i = w - fwd_i - 1
    fwd_dscores = (np.roll(np.array(list(genome)),-fwd_i) == rel_fwd_base) * dw
    rev_dscores = (np.roll(np.array(list(genome)),-rev_i) == rel_rev_base) * dw
    return fwd_scores + fwd_dscores,rev_scores + rev_dscores

def test_update_scores_np():
    w = 10
    matrix = random_energy_matrix(w)
    i = random.randrange(w)
    j = random.randrange(4)
    dw = random.random()
    print i,j,dw
    new_matrix = [row[:] for row in matrix]
    new_matrix[i][j] += dw
    genome = random_site(100000)
    fwd_scores,rev_scores = score_genome_np(matrix,genome,both_strands=True)
    fwd_scores_ref,rev_scores_ref = score_genome_np(new_matrix,genome,both_strands=True)
    fwd_scores_test,rev_scores_test = update_scores_np(fwd_scores,rev_scores,i,j,dw,w,genome)
    fwd_scores_ref2 = update_scores_np_ref(fwd_scores,i,j,dw,genome,w)
    print l2_np(fwd_scores_ref,fwd_scores_test),l2_np(fwd_scores_ref,fwd_scores_ref2)
    print l2_np(rev_scores_ref,rev_scores_test)
    
    
def l2_np(xs,ys):
    return np.sum((xs-ys)**2)
