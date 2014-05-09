import random
from math import log
from data import *
from project_utils import *
from chip_seq import *
from simulate_tf import theoretical_probabilities
from energy_matrix import random_energy_matrix
from metropolis_hastings import *

def log_likelihood(fragments,genome,energy_matrix):
    true_probs = theoretical_probabilities(energy_matrix,genome)
    binding_probs = [sum(true_probs[start:stop]) for (start,stop) in fragments]
    return sum(log(binding_prob)
               for binding_prob in binding_probs)

def sample_posterior(iterations=10000,num_fragments=10000,sigma=2,fragments=None):
    if fragments is None:
        print "Generating fragments"
        fragments = chip_seq_fragments(TRUE_ENERGY_MATRIX,GENOME,num_fragments)
    def logf(matrix):
        return log_likelihood(fragments,GENOME,matrix)
    def proposal(matrix):
        new_matrix = [row[:] for row in matrix] # make a copy of the matrix
        altered_col = random.randrange(W) # pick a column to alter
        altered_row = random.randrange(4) # pick a row to alter
        new_matrix[altered_col][altered_row] += random.gauss(0,sigma) # add N(0,2) noise
        return new_matrix
    init_matrix = random_energy_matrix(W)
    print "true log_likelihood:",log_likelihood(fragments,GENOME,TRUE_ENERGY_MATRIX)
    matrix_chain = mh(logf,proposal,init_matrix,iterations=iterations,use_log=True,verbose=10)
    return matrix_chain,fragments
