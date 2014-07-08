from sample import direct_sampling,rsa
from project_utils import score_seq,sample_average,inverse_cdf_sampler,falling_fac
from utils import random_site,pairs,mh,maybesave,transpose,product
from matplotlib import pyplot as plt
from math import log,exp
import random
random.seed(1)

genome = "ACGTTGCA" * 5 + random_site(80) + "ACGTTGCA" * 5 + random_site(10)
G = len(genome)
beta = 1
energy_matrix = [[-2,0,0,0],
                 [-0,-2,0,0],
                 [-0,0,-2,0],
                 [-0,0,0,-2],
                 [-0,0,0,-2],
                 [-0,0,-2,0],
                 [-0,-2,0,0],
                 [-2,0,0,0]]

w = len(energy_matrix)
config_len = G-w
interaction_energy = -8 # TFs in contact get -2 added to configuration energy
exclusion_energy = 1000000
eps =[score_seq(energy_matrix,genome[i:i+w]) for i in range(G-w+1)]
ks = [exp(-ep) for ep in eps]
    
def positions(config):
    return [i for i,x in enumerate(config) if x > 0]

def from_positions(poses):
    return [int(i in poses) for i in range(config_len)]
    
def hamiltonian(config):
    """Given a configuration describing the left-endpoints of tfs, compute
    associated energy"""
    poses = positions(config)
    total_site_energy = sum(score_seq(energy_matrix,genome[pos:pos+w])
                            for pos in poses)
    total_interaction_energy = interaction_energy * len([(i,j) for (i,j) in pairs(poses) if j - i == w])
    total_exclusion_energy = exclusion_energy * len([(i,j) for (i,j) in pairs(poses) if j - i < w])
    return total_site_energy + total_interaction_energy + total_exclusion_energy
    
def mh_simulate(iterations=50000,verbose=False,method="direct_sampling"):
    copy_number = 5
    def logf(config):
        return -hamiltonian(config)
    def prop(config):
        new_config = config[:]
        attached_tfs = sum(config) # number currently bound to chromosome
        r = random.random()
        if r < attached_tfs/float(copy_number): # choose a tf on the chromosome
            pos = random.choice(positions(config))
            new_config[pos] = 0
        # else: choose a tf off the chromosome
        new_pos = random.choice(range(config_len + 1))
        if new_pos < config_len:
            new_config[new_pos] = 1
        # else tf goes off chromosome
        return new_config
    Z = float(sum(ks))
    ps = [k/Z for k in ks]
    sampler = inverse_cdf_sampler(range(len(ks)),ps)
    def prop_direct(config):
        sample = direct_sampling(ks,copy_number,sampler=sampler)
        return from_positions(sample)
    def log_dprop_direct(config,old_config):
        occupancy = sum(config)
        poses = positions(config)
        return log(falling_fac(copy_number,occupancy)*product(exp(-beta*eps[i]*config[i])
                                                              for i in range(config_len)))
    def prop_rsa(config):
        sample = rsa(ks,copy_number)
        return from_positions(sample)
    def log_dprop_rsa(config,old_config):
        #print config
        _ks = ks[:]
        prob = 1
        for i,x in enumerate(config):
            if x > 0:
                prob *= _ks[i]/sum(_ks)
                #print x,prob
                _ks[i] = 0
        return log(prob)
    x0 = [0]*config_len
    if method == "direct_sampling":
        return mh(logf,prop_direct,x0,dprop=log_dprop_direct,verbose=verbose,use_log=True,iterations=iterations)
    elif method == "rsa":
        return mh(logf,prop_rsa,x0,dprop=log_dprop_rsa,verbose=verbose,use_log=True,iterations=iterations)
    else:
        return mh(logf,prop,x0,dprop=None,verbose=verbose,use_log=True,iterations=iterations)

def acceptance_ratio(sample):
    ars = [0.0]
    acceptances = 0.0
    for n,(x,y) in enumerate(pairs(sample)):
        if x != y:
            acceptances += 1
        ars.append(acceptances/(n+1))
    return ars
    
def viz_sample(sample,filename=None):
    """Visualize a sample trajectory"""
    plt.subplot(211)
    plt.imshow(transpose(sample),interpolation='nearest',aspect='auto')
    plt.ylabel("Position")
    plt.subplot(212)
    energies = map(hamiltonian,sample)
    plt.plot(energies)
    plt.ylabel("Energy")
    plt.xlabel("Iteration")
    maybesave(filename)


print "loaded"
