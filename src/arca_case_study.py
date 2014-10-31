"""
In this script we attempt to recover the binding energy model of ArcA
"""

import sys
sys.path.append("../data/chip_seq_datasets")
from viz_map import read_map,density_from_reads
from energy_matrix import random_energy_matrix,score_genome_np,update_scores_np
from chip_seq import reads_from_ps,density_from_ps
from fd import fd_solve_np
from utils import sample,get_ecoli_genome,mh,pprint,make_pssm,wc,sorted_indices,rslice
from project_utils import np_normalize,log_dbinom,np_log_fac,timestamp
import numpy as np
from math import log
from fd_inference import capture_state,log_dprop,complete_rprop,MU_SIGMA,MAT_SIGMA
from motifs import Escherichia_coli

genome_filename = "/home/pat/chip_seq_inference/data/chip_seq_datasets/ArcA_park_et_al/SRR835423/U00096.2.fna"
with open(genome_filename) as f:
    lines = f.readlines()
genome = "".join([line.strip() for line in lines[1:]])
G = len(genome)

def arca_motif_comparison():
    arca_reads = get_arca_reads(1000000)
    true_rdm = density_from_reads(arca_reads,G)
    pssm = make_pssm(Escherichia_coli.ArcA)
    plt.plot(true_rdm[0])
    plt.plot(true_rdm[1])
    fwd_scores = score_genome_np(pssm,genome)
    rev_scores = score_genome_np(pssm,wc(genome))
    scores = np.log(np.exp(fwd_scores) + np.exp(rev_scores))
    sites = concat([(site,wc(site)) for site in Escherichia_coli.ArcA])
    site_locations = [m.start(0) for site in sites for m in re.finditer(site,genome)]
    site_locations_np = np.zeros(G)
    for site_loc in site_locations:
        site_locations_np[site_loc] = 1
    plt.plot(site_locations_np*100)
    plt.plot()
    plt.plot(scores)

def cumsum_test():
    arca_reads = get_arca_reads(1000000)
    true_rdm = density_from_reads(arca_reads,G)
    pssm = make_pssm(Escherichia_coli.ArcA)
    comb_rdm = true_rdm[0] + true_rdm[1]
    print "fwd_scores"
    fwd_scores = score_genome_np(pssm,genome)
    print "rev_scores"
    rev_scores = score_genome_np(pssm,wc(genome))
    scores = np.log(np.exp(fwd_scores) + np.exp(rev_scores))
    probs = np.exp(scores)/np.sum(np.exp(scores))
    print "sorting scores"
    score_js = sorted_indices(scores)[::-1] # order scores from greatest to least
    print "sorting probs"
    prob_js = sorted_indices(probs)[::-1] # ditto
    plt.plot(cumsum(rslice(comb_rdm,score_js)),label="scores")
    plt.plot(cumsum(rslice(comb_rdm,prob_js)),label="boltzmann probs")
    comb_rdm_copy = list(comb_rdm)
    controls = 5
    for i in range(controls):
        print i
        random.shuffle(comb_rdm_copy)
        plt.plot(cumsum(comb_rdm_copy),color='r')
    plt.legend(loc=0)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

def gc_bias_test():
    arca_reads = get_arca_reads(1000000)
    true_rdm = density_from_reads(arca_reads,G)
    comb_rdm = true_rdm[0] + true_rdm[1]
    gc_pssm = [[0,1,1,0]] * 10
    gc_scores = score_genome_np(gc_pssm,genome)
    plt.scatter(gc_scores,comb_rdm,marker='.')
    
def get_arca_reads(N=100000):
    """Return N downsampled reads from ArcA dataset"""
    filename = '/home/pat/chip_seq_inference/data/chip_seq_datasets/ArcA_park_et_al/SRR835423/SRR835423.map'
    arca_reads = read_map(filename)
    sampled_arca_reads = sample(N,arca_reads) # Downsample for numerical convenience...
    sampled_read_fraction = len(sampled_arca_reads)/float(len(arca_reads))
    print "sampled %1.2f%% of %s reads" % (sampled_read_fraction*100,len(arca_reads))
    return sampled_arca_reads

def rdm_log_likelihood(true_rdm,proposed_rdm):
    """Compute the likelihood of true RDM (read density map), given proposed RDM.  

    RDM consists of two density maps: fwd and rev.  Maps should be
    unnormalized, such that xs[i] contains number of reads which map
    into ith base.  For each strand, convert proposed_rdm to
    probabilities such that proposed_rdm contains probability that
    base i is covered by an arbitrary read.  Then compute likelihood,
    treating each base as independent binomial random variable.
    """
    ks_fwd,ks_rev = true_rdm
    ps_fwd,ps_rev = [np_normalize(rdm+1) for rdm in proposed_rdm]
    fwd_N,rev_N = np.sum(ks_fwd),np.sum(ks_rev)
    #fwd_ll = sum(log_dbinom_approx(k,fwd_n,p) for (k,p) in zip(true_fwd,p_fwd))
    fwd_ll = np.sum(ks_fwd*log(fwd_N) - np_log_fac(ks_fwd)
                    + ks_fwd*np.log(ps_fwd) + (fwd_N-ks_fwd)*np.log(1-ps_fwd))
    #rev_ll = sum(log_dbinom_approx(k,rev_n,p) for (k,p) in zip(true_rev,p_rev))
    rev_ll = np.sum(ks_rev*log(rev_N) - np_log_fac(ks_rev)
                    + ks_rev*np.log(ps_rev) + (rev_N-ks_rev)*np.log(1-ps_rev))
    return fwd_ll + rev_ll

def rdm_log_likelihood_spec(true_rdm,proposed_rdm):
    """Compute the likelihood of true RDM (read density map), given proposed RDM, by pearson correlation"""
    ks_fwd,ks_rev = true_rdm
    ks_comb = ks_fwd + ks_rev
    ps_fwd,ps_rev = proposed_rdm
    ps_comb = ps_fwd + ps_rev
    #return -(np.sum(ks_fwd-ps_fwd)**2 + np.sum(ks_rev-ps_rev)**2)
    fwd_cor = pearsonr(ks_fwd,ps_fwd)[0]
    rev_cor = pearsonr(ks_rev,ps_rev)[0]
    comb_cor = pearsonr(ks_comb,ps_comb)[0]
    print "correlations:",fwd_cor,rev_cor,comb_cor
    #return log(fwd_cor+1) + log(fwd_cor+1)
    return log(comb_cor + 1)
    
def complete_log_likelihood(state,true_rdm,lamb,num_reads=100000):
    """Compute log likelihood of true_rdm given energy model (state).
    
    (1) Simulate reads from energy model.
    (2) compare simulated reads to true reads with read_log_likelihood."""
    print "num_reads:",num_reads,"%e" % num_reads
    (matrix,mu),all_eps = state
    ps = fd_solve_np(all_eps,mu)
    G = len(ps)
    MFL = 1/lamb
    print "generating reads"
    proposed_reads = reads_from_ps(ps,MFL,min_seq_len=75,num_reads=num_reads)
    print "mapping reads"
    proposed_rdm = density_from_reads(proposed_reads,G)
    #proposed_rdm = density_from_ps(ps,MFL,min_seq_len=75,num_reads=num_reads)
    return rdm_log_likelihood(true_rdm,proposed_rdm)

def rdm_from_state((matrix,mu),num_reads=100000,eps=None):
    if eps is None:
        print "scoring genome"
        eps = score_genome_np(matrix,genome)
    print "solving ps"
    ps = fd_solve_np(eps,mu)
    print "generating reads"
    reads = reads_from_ps(ps,MFL,min_seq_len=75,num_reads=num_reads)
    print "mapping reads"
    rdm = density_from_reads(reads,G)
    return rdm

def plot_state(state,num_reads=100000):
    a,b = state
    if type(a) is tuple: # if state consists of ((matrix,mu),eps)
        matrix,mu = a
        eps = b
        rdm = rdm_from_state((matrix,mu),num_reads,eps)
    else:
        matrix,mu = a,b
        eps = score_genome_np(matrix,genome)
        rdm = rdm_from_state((matrix,mu),num_reads,eps)
    plt.plot(rdm[0])
    plt.plot(rdm[1])
        
def infer_arca_energy_model(num_reads=100000):
    """the whole show: infer the energy model from true reads"""
    true_reads = get_arca_reads(num_reads)
    G = len(genome)
    MFL = 250 #mean frag length = 250bp
    lamb = 1/250.0
    true_rdm = density_from_reads(true_reads,G)
    w = 10
    init_matrix = random_energy_matrix(w)
    init_mu = -20
    init_scores = score_genome_np(init_matrix,genome)
    init_state = ((init_matrix,init_mu),init_scores)
    logf = lambda state:timestamp(complete_log_likelihood(state,true_rdm,lamb,num_reads))
    rprop = lambda state:complete_rprop(state,genome)
    verbose = True
    iterations = 50000
    matrix_chain = mh(logf,proposal=rprop,x0=init_state,dprop=log_dprop,
                      capture_state=capture_state,verbose=verbose,
                      use_log=True,iterations=iterations,modulus=100)
    return matrix_chain

def infer_synthetic_energy_model(num_reads=100000):
    """the whole show: infer the energy model from true reads"""
    genome = get_ecoli_genome(at_lab=False)
    G = len(genome)
    w = 10
    true_matrix = [[-2,0,0,0] for i in range(w)]
    true_mu = -20
    true_eps = score_genome_np(true_matrix,genome)
    true_ps = fd_solve_np(true_eps,true_mu)
    MFL = 250 #mean frag length = 250bp
    lamb = 1/250.0
    true_reads = reads_from_ps(true_ps,MFL,min_seq_len=75,num_reads=num_reads)
    true_rdm = density_from_reads(true_reads,G)
    init_matrix = random_energy_matrix(w)
    init_mu = -20
    init_scores = score_genome_np(init_matrix,genome)
    init_state = ((init_matrix,init_mu),init_scores)
    logf = lambda state:timestamp(complete_log_likelihood(state,true_rdm,lamb,num_reads=num_reads))
    rprop = lambda state:complete_rprop(state,genome)
    verbose = True
    iterations = 50000
    print "true_ll:",logf(((true_matrix,true_mu),true_eps))
    matrix_chain = mh(logf,proposal=rprop,x0=init_state,dprop=log_dprop,
                      capture_state=capture_state,verbose=verbose,
                      use_log=True,iterations=iterations,modulus=100)
    return matrix_chain

def gradient_descent_experiment(true_rdm=None,num_reads=100000):
    genome = get_ecoli_genome(at_lab=False)
    G = len(genome)
    w = 10
    mfl = 250
    lamb = 1.0/mfl
    if true_rdm is None:
        true_matrix = [[-2,0,0,0] for i in range(w)]
        true_mu = -20
        true_eps = score_genome_np(true_matrix,genome)
        true_ps = fd_solve_np(true_eps,true_mu)
        true_reads = reads_from_ps(true_ps,MFL,min_seq_len=75,num_reads=num_reads)
        true_rdm = density_from_reads(true_reads,G)
        true_state = ((true_matrix,true_mu),true_eps)
    true_ll = logf(true_state) if true_rdm is None else None
    matrix = random_energy_matrix(w)
    mu = -20
    eps = score_genome_np(matrix,genome)
    init_state = ((matrix,mu),eps)
    logf = lambda state:timestamp(complete_log_likelihood(state,true_rdm,lamb,num_reads=num_reads))
    dw = 0.1
    dmu = 0.1
    old_ll = 0
    print "true_ll:",true_ll
    cur_ll = logf(init_state)
    eta = 10**-7 # learning rate
    iterations = 0
    while cur_ll > old_ll or iterations == 0:
        old_ll = cur_ll
        dmat = [[0]*4 for i in range(w)]
        for i in range(w):
            for j in range(4):
                    print "i,j:",i,j
                    new_mat = [row[:] for row in matrix]
                    new_mat[i][j] += dw
                    new_eps = update_scores_np(eps,i,j,dw,w,genome)
                    new_state = ((new_mat,mu),new_eps)
                    new_ll = logf(new_state)
                    print "cur ll,new_ll:", cur_ll,new_ll,"(improvement)" if new_ll > cur_ll else "(worsening)"
                    delta_w = (new_ll - cur_ll)/dw * eta
                    print "delta_w:",delta_w
                    dmat[i][j] = delta_w
        new_mu = mu + dmu
        new_state = ((mat,new_mu),eps)
        new_ll = logf(new_state)
        print "mu:"
        print "cur ll,new_ll:", cur_ll,new_ll,"(improvement)" if new_ll > cur_ll else "(worsening)"
        delta_mu = (new_ll - cur_ll)/dmu * eta
        print "delta_mu:",delta_mu
        old_matrix = [row[:] for row in matrix]
        for i in range(w):
            for j in range(4):
                matrix[i][j] += dmat[i][j]
        old_eps = np.array(eps)
        eps = score_genome_np(matrix,genome)
        old_mu = mu
        mu += delta_mu
        cur_state = ((mat,mu),eps)
        cur_ll = logf(cur_state)
        print "\nresults of iteration %s:" % iterations
        pprint(matrix)
        print mu
        print "likelihood:",old_ll,"->",cur_ll
        iterations += 1
    return ((old_matrix,old_mu),old_eps)
