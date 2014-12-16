from sample_configs import *

from project_utils import random_site,subst
from energy_matrix import random_energy_matrix, score_genome_np
from utils import transpose

G = 50000
w = 10
mu = -20
epi = -10 #interaction energy
genome = random_site(G)
genome = subst(genome,"AAAAAAAAAA",100)
genome = subst(genome,"AAAAAAAAAA",110)
genome = subst(genome,"AAAAAAAAAA",120)
energy_matrix = [[-2,0,0,0] for i in range(w)]
eps = score_genome_np(energy_matrix,genome)
eps = -np.log(np.exp(-eps[0]) + np.exp(-eps[1]))
interactions = [nearest_neighbor_interaction(int_energy,w)]
print "setup done"

def test_fd():
    fd_chain = lift_from_fd(eps,mu,interactions)
    return fd_chain
    
def test_rsa():
    rsa_chain = lift_from_rsa(eps,mu,w,epi,interactions)
    return rsa_chain

def test_rsa_with_prob():
    """do rsa_with_prob and rsa_with_prob2 agree?"""
    trials = 100
    configs_with_probs = [rsa_with_prob(eps,mu,w,epi,100) for _ in verbose_gen(xrange(trials))]
    configs_with_probs2 = [rsa_with_prob2(eps,mu,w,epi,100) for _ in verbose_gen(xrange(trials))]
    configs,probs = transpose(configs_with_probs)
    configs2,probs2 = transpose(configs_with_probs2)
    profile = mean(configs)
    profile2 = mean(configs2)
    plt.plot(profile)
    plt.plot(profile2)
