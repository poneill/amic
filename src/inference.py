import random
from math import log
from data import *
from project_utils import *
from chip_seq import *
from simulate_tf import theoretical_probabilities
from energy_matrix import *
from metropolis_hastings import *
from utils import maybesave

def log_likelihood(fragments,site_scores):
    site_probs = scores_to_probs(site_scores)
    binding_probs = [sum(site_probs[start:stop]) for (start,stop) in fragments]
    return sum(log(binding_prob)
               for binding_prob in binding_probs)

def sample_posterior(iterations=10000,num_fragments=10000,sigma=1,fragments=None):
    if fragments is None:
        fragments = chip_seq_fragments(TRUE_ENERGY_MATRIX,GENOME,num_fragments)
    def logf((matrix,site_scores)):
        ll = log_likelihood(fragments,site_scores)
        return ll
    def proposal((matrix,site_scores)):
        pprint(matrix)
        new_matrix = [row[:] for row in matrix] # make a copy of the matrix
        altered_col = random.randrange(W) # pick a column to alter
        altered_row = random.randrange(4) # pick a row to alter
        dw = random.gauss(0,sigma) # add N(0,2) noise
        new_matrix[altered_col][altered_row] += dw
        new_scores = update_scores(site_scores,altered_col,altered_row,dw)
        return (new_matrix,new_scores)
    def capture_state((matrix,site_scores)):
        return matrix
    init_matrix = random_energy_matrix(W)
    init_state = (init_matrix,score_genome(init_matrix,GENOME))
    true_site_scores = score_genome(TRUE_ENERGY_MATRIX,GENOME)
    print "true log_likelihood:",log_likelihood(fragments,true_site_scores)
    matrix_chain = mh(logf,proposal,init_state,iterations=iterations,use_log=True,
                      capture_state=capture_state,verbose=1)
    return matrix_chain,fragments

def plot_matrix_chain(mc,true_ll,filename=None):
    lls = [m[1] for m in mc]
    plt.plot(lls)
    plt.plot([true_ll for l in lls],label='True Log-likelihood',linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.legend(loc='lower right')
    maybesave(filename)

def map_fragments(fragments):
    arr = [0] * len(GENOME)
    for (start,stop) in fragments:
        for i in range(start,stop):
            arr[i]+=1
    return arr

def plot_fragments(fragments,filename=None):
    plt.plot(map_fragments(fragments))
    plt.xlabel("Genomic coordinate")
    plt.ylabel("Read Density")
    plt.title("Simulated ChIP-Seq Read Map")
    maybesave(filename)

def plot_energy_matrix(matrix,filename=None):
    plt.imshow(transpose([[x - max(row) for x in row] for row in matrix]),
               interpolation='nearest')
    plt.colorbar()
    maybesave(filename)
    
