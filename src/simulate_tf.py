import random
from math import exp
from energy_matrix import score_genome
from metropolis_hastings import mh

def simulate(energy_matrix,genome,steps):
    """Return genomic position of TF with given energy matrix after given
    number of steps."""
    w = len(energy_matrix) # width of binding sites, in bases
    L = len(genome)
    binding_landscape = score_genome(energy_matrix,genome)
    def binding_probability(i):
        """Return probability, up to a constant, of occupying ith position in
        genome"""
        delta_G = binding_landscape[i]
        return exp(-delta_G)
    def diffusion3d(i):
        """Choose a position at random"""
        return random.randrange(L-w+1)
    return mh(p=binding_probability,proposal=diffusion3d,x0=0,iterations=steps)

def theoretical_probabilities(energy_matrix,genome):
    binding_landscape = score_genome(energy_matrix,genome)
    Z = float(sum([exp(-score) for score in binding_landscape]))
    return [exp(-score)/Z for score in binding_landscape]
    
    
    
