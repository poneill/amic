"""
In this script we attempt to recover a binding energy model with interaction energies
"""
from math import exp
import numpy as np

from utils import random_site,pprint,dnorm,mh,maybesave
from project_utils import subst,timestamp
from energy_matrix import score_genome_np,random_energy_matrix,update_scores_np
from sample_configs import nearest_neighbor_interaction,lift_from_fd,lift_from_rsa
from chip_seq import reads_from_ps
from arca_case_study import rdm_log_likelihood
import random
from math import log,exp
import sys
sys.path.append("../data/chip_seq_datasets")
from viz_map import density_from_reads
from frags_from_reads import frag_sabot

MAT_SIGMA = 1
MU_SIGMA = 0.1
EPI_SIGMA = 1
mfl = 250 # mean frag len
msl = 75 # min seq len
lamb = 1.0/mfl

def make_synthetic_genome(G,motif):
    genome = random_site(G)
    w = len(motif)
    genome = subst(genome,motif,100)
    genome = subst(genome,motif,100+w)
    genome = subst(genome,motif,100+2*w)
    return genome

def energy_matrix_recognizing(motif):
    def base_to_row(b):
        return [-2*(b==c) for c in "ACGT"]
    return [base_to_row(b) for b in motif]
    
def main():
    num_true_reads=1000000
    num_prop_reads=10000
    num_true_configs = 10000
    num_prop_configs = 1000
    ol_iterations = 50000
    # True params
    G = 50000
    motif = "ACGTTGCA"
    w = len(motif)
    genome = make_synthetic_genome(G,motif)
    true_mu = -15
    true_epi = -10 #interaction energy
    true_matrix = energy_matrix_recognizing(motif)
    interactions = [nearest_neighbor_interaction(true_epi,w)]
    true_eps = score_genome_np(true_matrix,genome)
    comb_eps = combine_eps(true_eps)
    # end of true params
    # set up dataset
    mean_config = lift_from_fd(comb_eps,true_mu,w,true_epi,interactions,iterations=num_true_configs,verbose=False)
    true_reads = reads_from_ps(mean_config,mfl,msl,num_reads=num_true_reads)
    true_rdm = density_from_reads(true_reads,G)
    # set up init_state
    init_matrix = null_energy_matrix(w)
    init_mu = true_mu
    init_epi = 0
    init_eps = score_genome_np(init_matrix, genome)
    init_state = ((init_matrix, init_mu, init_epi), init_eps)
    true_state = ((true_matrix, true_mu, true_epi), true_eps)
    logf = lambda state:timestamp(complete_log_likelihood(state, true_rdm, lamb, num_prop_configs,num_prop_reads))
    rprop = lambda state:complete_rprop(state, genome)
    verbose = True
    print "true log-likelihood:",logf(true_state)
    bem_chain = mh_with_prob(logf, proposal=rprop, x0=init_state, dprop=log_dprop, 
                      capture_state=capture_state, verbose=verbose, 
                      use_log=True, iterations=ol_iterations, modulus=100,cache=False)
    return bem_chain

def complete_log_likelihood(state, true_rdm, lamb,il_iterations=10000,num_reads=100000):
    """Compute log likelihood of true_rdm given energy model (state).
    
    (1) Simulate reads from energy model.
    (2) compare simulated reads to true reads with read_log_likelihood."""
    print "num_reads:", num_reads, "%e" % num_reads
    print len(state),map(len,state)
    (matrix, mu, epi), all_eps = state
    G = len(all_eps[0]) # since fwd and rev energies...
    w = len(matrix)
    print "generating reads"
    interactions = [nearest_neighbor_interaction(epi,w)]
    comb_eps = combine_eps(all_eps)
    mean_config = lift_from_fd(comb_eps,mu,w,epi,interactions,iterations=il_iterations,verbose=False)
    proposed_reads = reads_from_ps(mean_config,mfl,msl,num_reads=num_reads)
    print "mapping reads"
    proposed_rdm = density_from_reads(proposed_reads, G)
    #proposed_rdm = density_from_ps(ps, MFL, min_seq_len=75, num_reads=num_reads)
    return rdm_log_likelihood(true_rdm, proposed_rdm)

def complete_rprop(((mat,mu,epi),(fwd_eps,rev_eps)),genome):
    """Propose a new matrix and new mu, given mat,mu.  Return updated
    scores and probability of proposal for convenience.  Taken from fd_inference, updated to
    include interaction energy.

    """
    pprint(mat)
    print "mu:",mu
    print "epi:",epi
    w = len(mat)
    new_mat = [row[:] for row in mat] # make a copy of the matrix
    new_mu = mu
    new_epi = epi
    r = random.random()
    if r < 1/3.0: # flip a coin and update weight matrix or mu
        if r < 1/3.0*.98:
            altered_col = random.randrange(w) # pick a column to alter
            altered_row = random.randrange(4) # pick a row to alter
            dw = random.gauss(0,MAT_SIGMA) # add N(0,2) noise
            new_mat[altered_col][altered_row] += dw
            log_p = log(1/3.0*.98*dnorm(dw,0,MAT_SIGMA))
            new_fwd_eps,new_rev_eps = update_scores_np(fwd_eps,rev_eps,altered_col,altered_row,dw,w,genome)
        else: #shift
            print "shifting weight matrix..."
            if r < 1/3.0*.99: # shift forward
                new_mat = [new_mat[-1]] + new_mat[:-1]
            else: # shift_backwards
                new_mat = new_mat[1:] + [new_mat[0]]
            new_fwd_eps,new_rev_eps = score_genome_np(new_mat,genome)
            log_p = log(1/3.0*.01*dnorm(dw,0,MAT_SIGMA))
    elif r < 2/3.0:
        dmu = random.gauss(0,MU_SIGMA)
        new_mu += dmu
        new_fwd_eps,new_rev_eps = fwd_eps,rev_eps # careful about returning copy...?
        log_p = log(1/3.0*dnorm(dnmu,0,MU_SIGMA))
    else:
        depi = random.gauss(0,EPI_SIGMA)
        new_epi += depi
        new_fwd_eps,new_rev_eps = fwd_eps,rev_eps # careful about returning copy...?
        log_p = log(1/3.0*dnorm(depi,0,EPI_SIGMA))
    return ((new_mat,new_mu,new_epi),(new_fwd_eps,new_rev_eps)),log_p

def log_dprop(((matp,mup,epip),epsp),((mat,mu,epi),eps)):
    dmat = sum([xp - x for (rowp,row) in zip(matp,mat) for (xp,x) in zip(rowp,row)])
    dmu = mup - mu
    depi = epip - epi
    if dmat != 0:
        return log(1/3.0 * dnorm(dmat,0,MAT_SIGMA))
    elif dmu != 0:
        return log(1/3.0 * dnorm(dmu,0,MAT_SIGMA))
    else: # depi != 0
        return log(1/3.0 * dnorm(depi,0,EPI_SIGMA))
        #return log(dnorm(dmat,0,MAT_SIGMA)) + log(dnorm(dmu,0,MU_SIGMA))

def combine_eps(all_eps):
    """Convert fwd, rev eps to ss eps"""
    return -np.log(np.exp(-all_eps[0]) + np.exp(-all_eps[1]))

def capture_state((state,eps)):
    return state

def figure1(filename=None,linewidth=1):
    # fig = pl.figure(1)
    # plot = fig.add_subplot(111)
    plt.plot(true_rdm[0],color='r',label="Synthetic Data",linewidth=linewidth)
    plt.plot(true_rdm[1],color='r',linewidth=linewidth)
    plt.plot(init_rdm[0],color='b',label="Initial Model",linewidth=linewidth)
    plt.plot(init_rdm[1],color='b',linewidth=linewidth)
    plt.plot(rec_rdm[0],color='g',label="Recovered Model",linewidth=linewidth)
    plt.plot(rec_rdm[1],color='g',linewidth=linewidth)
    plt.xlim(37500,40500)
    plt.ylim(0,4000)
    plt.legend(loc='upper right',prop={'size':18})
    plt.xlabel("Genomic Coordinate",fontsize=20)
    plt.ylabel("Read Density",fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    #plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig(filename)
    plt.close()
    #maybesave(filename)
    
