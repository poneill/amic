"""
Functions for generating chip seq datasets
"""
import random
from project_utils import *
from data import *
from simulate_tf import theoretical_probabilities

def chip_seq_fragments(energy_matrix,genome,num_fragments):
    """Simulate a chIP-seq dataset for energy_matrix on genome containing
    given number of fragments.  Return a list of tuples containing
    (left_endpoint,right_endpoint) of fragments,
    """
    fragments = []
    true_probs = theoretical_probabilities(energy_matrix,genome)
    for i in xrange(num_fragments):
        # Determine position of TF on genome
        pos = inverse_cdf_sample(range(L-W+1),true_probs)
        # Determine length of fragment containing TF
        frag_length = rpois(MEAN_FRAG_LENGTH)
        # Determine distance from TF to left-endpoint of fragment
        offset = min(random.randrange(frag_length),pos)
        # collect left, right endpoints of fragment in list of fragments
        fragments.append((pos-offset,pos-offset+frag_length))
    return fragments

