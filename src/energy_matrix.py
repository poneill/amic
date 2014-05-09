"""Utility functions for energy matrices"""

import random

# The 0th row is A, the 1st row is C...
base_index = dict(zip("ACGT",range(4)))

def random_energy_matrix(width,max_val=1):
    """Return a energy matrix of given width containing entries uniformly
    distributed in [-max_val,0] """
    return [[-max_val*random.random() for base in "ACGT"] for i in range(width)]

def score(energy_matrix,sequence):
    """Compute the score of sequence with energy matrix"""
    return sum(column[base_index[base]] for (base, column) in zip(sequence,energy_matrix))

def score_genome(energy_matrix,genome):
    """Score entire genome, returning a list of delta G's indexed by left
    endpoint.  If genome is of length L and matrix of width w, binding
    landscape is of length L-w+1"""
    L = len(genome)
    w = len(energy_matrix)
    return [score(energy_matrix,genome[i:i+w]) for i in range(L-w+1)]
