"""Try a full-scale energy matrix recovery using an FD model for both
true and assumed dynamics."""
from project_utils import random_site,leps_from_config,pprint
from utils import bisect_interval,verbose_gen,concat,product,dnorm,mh,show,pprint
from fd import fd_solve_np
from fd import rfd_xs,rfd_xs_np
from math import log,exp
from chip_seq import chip_ps,chip_ps_np,map_reads,map_reads_np,predict_chip_ps4,chip_seq_log_likelihood
from energy_matrix import random_energy_matrix,score_genome_np,update_scores_np
from scipy.stats import pearsonr
import numpy as np
import random
import signal
import sys

w = 10
q = 100
beta = 1
TRUE_ENERGY_MATRIX = [[-2,0,0,0] for i in range(w)]
NUM_CELLS_ORIGINAL = 10000
NUM_CELLS_RECOVERED = 10000
MEAN_FRAGMENT_LENGTH=50
MAT_SIGMA = 1
MU_SIGMA = 0.1

def complete_rprop(((mat,mu),(fwd_eps,rev_eps)),genome):
    """Propose a new matrix and new mu, given mat,mu.  Return updated
    scores for convenience"""
    pprint(mat)
    print "mu:",mu
    w = len(mat)
    new_mat = [row[:] for row in mat] # make a copy of the matrix
    new_mu = mu
    if random.random() < 0.5: # flip a coin and update weight matrix or mu
        altered_col = random.randrange(w) # pick a column to alter
        altered_row = random.randrange(4) # pick a row to alter
        dw = random.gauss(0,MAT_SIGMA) # add N(0,2) noise
        new_mat[altered_col][altered_row] += dw
        new_fwd_eps,new_rev_eps = update_scores_np(fwd_eps,rev_eps,altered_col,altered_row,dw,w,genome)
    else:
        new_mu += random.gauss(0,MU_SIGMA)
        new_fwd_eps,new_rev_eps = fwd_eps,rev_eps # careful about returning copy...?
    return ((new_mat,new_mu),(new_fwd_eps,new_rev_eps))

def log_dprop(((matp,mup),epsp),((mat,mu),eps)):
    dmat = sum([xp - x for (rowp,row) in zip(matp,mat) for (xp,x) in zip(rowp,row)])
    dmu = mup - mu
    if dmat != 0:
        return log(1/2.0 * dnorm(dmat,0,MAT_SIGMA))
    else:
        return log(1/2.0 * dnorm(dmu,0,MAT_SIGMA))
        #return log(dnorm(dmat,0,MAT_SIGMA)) + log(dnorm(dmu,0,MU_SIGMA))
    
def capture_state((mat_and_mu,site_scores)):
    return mat_and_mu

def complete_log_likelihood(((matrix,mu),eps),mapped_reads,num_cells=NUM_CELLS_RECOVERED):
    """Compute log likelihood of matrix, given chip seq data"""
    print "entering complete log likelihood"
    ps = np.append(fd_solve_np(eps,mu),[0]*(w-1))
    G = len(ps)
    #print "G=",G
    # if random.random() < 1:#0.01:
    #     pprint(matrix)
    print "mean copy number:",np.sum(ps),"mu:",mu
    #print "predicting mapped_reads"
    #predicted_coverage_probability = predict_chip_ps4(ps,MEAN_FRAGMENT_LENGTH,1) # XXX HACK
    proposed_reads = map_reads_np(chip_ps_np(ps,MEAN_FRAGMENT_LENGTH,num_cells),G)
    #print "predicted mapped_reads"
    # add laplacian pseudocount: one observation of hit and miss each
    predicted_coverage_probability = (np.array(proposed_reads,dtype=float)+1)/(num_cells+2)
    #print "computing likelihood"
    #print "pearson correlation between true, recovered datasets:",pearsonr(proposed_reads,mapped_reads)
    ans = chip_seq_log_likelihood(predicted_coverage_probability,mapped_reads,NUM_CELLS_ORIGINAL)
    if True:#random.random() < 0.01:
        pprint(matrix)
        print "mu:",mu
        print "log likelihood:",ans
        print "returning from complete log likelihood"
    return ans

def log_likelihood_from_state(matrix,mu,genome,mapped_reads,num_cells=NUM_CELLS_RECOVERED):
        eps = score_genome_np(matrix,genome)
        return complete_log_likelihood(((matrix,mu),eps),mapped_reads,num_cells=num_cells)
        
def main(G=5000000,iterations=50000,init_matrix=None,init_mu=None,verbose=True):
    """Test case for FD-inference"""
    print "generating genome"
    genome = random_site(G)
    print "generating eps"
    eps = score_genome_np(TRUE_ENERGY_MATRIX,genome)
    min_mu,max_mu = -40,0
    mu = bisect_interval(lambda mu:np.sum(fd_solve_np(eps,mu))-q,min_mu,max_mu,verbose=True,tolerance=1e-1)
    print "computing ps"
    true_ps = fd_solve_np(eps,mu)
    print "true q:",np.sum(true_ps)
    print "generating chip dataset"
    mapped_reads = np.array(map_reads_np(chip_ps_np(true_ps,MEAN_FRAGMENT_LENGTH,NUM_CELLS_ORIGINAL),G))
    print "finished chip dataset"
    if init_matrix is None:
        init_matrix = random_energy_matrix(w)
    if init_mu is None:
        init_mu = -20#random.random()*40 - 20
    init_scores = score_genome_np(init_matrix,genome)
    init_state = ((init_matrix,init_mu),init_scores)
    logf = lambda state:complete_log_likelihood(state,mapped_reads)
    print "true mu:",mu
    print "true log_likelihood:",logf(((TRUE_ENERGY_MATRIX,mu),eps))
    rprop = lambda state:complete_rprop(state,genome)
    print "hitting mh loop"
    matrix_chain = mh(logf,proposal=rprop,x0=init_state,dprop=log_dprop,capture_state=capture_state,verbose=verbose,use_log=True,iterations=iterations,modulus=100)
    return matrix_chain,genome,mapped_reads

def compare(good_mat,good_mu,true_mat,true_mu,genome):
    true_eps = score_genome_np(true_mat,genome)
    good_eps = score_genome_np(good_mat,genome)
    true_ps = fd_solve_np(true_eps,true_mu)
    good_ps = fd_solve_np(good_eps,good_mu)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(true_ps)
    ax1.plot(good_ps)
    #axarr[0].set_title('Sharing X axis')
    ax2 = fig.add_subplot(222)
    ax2.scatter(true_ps, good_ps)
    ax2.plot([0,1],[0,1])
    # ax2.xlim(0,1)
    # ax2.ylim(0,1)
    def offset(mat):
            """Normalize each column so that max element is zero"""
            return [[r-max(row) for r in row] for row in mat]
    ax3 = fig.add_subplot(223)
    cax3 = ax3.imshow(transpose(true_mat),interpolation='none')
    fig.colorbar(cax3)
    ax4 = fig.add_subplot(224)
    cax4 = ax4.imshow(transpose(offset(good_mat)),interpolation='none')
    fig.colorbar(cax4)
    #print "Pearson r:",pearsonr(true_ps,good_ps)
    plt.show()
    
def make_chip_dataset(num_cells):
    return concat([chip(genome,rfd_xs(ps),MEAN_FRAGMENT_LENGTH) for i in verbose_gen(xrange(num_cells))])

def read_likelihood_deprecated(read,ps):
    """Compute the likelihood of observing reads given ps.  What we want
to compute, specifically, is the likelihood of the read being bound by at least one TF.  WRONG"""
    start,stop = read
    return 1 - product(1-p for p in ps[start:stop])

def cell_likelihood(reads,ps):
    points = sorted(concat(reads))
    G = len(ps)
    if not 0 in points:
        points.append(0)
    if not G in points:
        points.append(G)
    read_complements = [(stop)]
    return product([product(1-p for p in ps[start:stop]) for (start,stop) in reads])

def prob_at_least_one(ps):
    return 1 - product(1-p for p in ps)

def rightward_prob_sanity_check(ps,lamb):
    """Compute the probability of rightward end of fragment being captured"""
    i = 0
    G = len(ps)
    while True:
        if random.random() < ps[i]:
            #print "succeeded at:",i
            return True
        if random.random() > lamb and i < G-1:
            i += 1
        else:
            #print "failed at:",i
            return False

def prob_end_at_i(i,lamb):
    return lamb*(1-lamb)**(i)
    
def rightward_prob_analytic(ps,lamb):
    return sum(prob_end_at_i(i,lamb)*prob_at_least_one(ps[:i+1]) for i in range(len(ps)))

def rightward_prob_analytic_fast(ps,lamb):
    success_prob = 0
    prob_this_far = 1
    for i,p in enumerate(ps):
        success_prob += prob_this_far*(1-lamb)**i*p
        prob_this_far *= (1-p)
    return success_prob
    
def point_prob(ps,i,mean_frag_length):
    """Compute probability that a random (captured) fragment will cover base i."""
    lamb = 1.0/mean_frag_length
    cutoff = 10*mean_frag_length
    G = len(ps)
    print "cutoff:",cutoff
    print "computing rightward"
    print "rightward ps:",len(ps[i:i+cutoff])
    rightward_prob = rightward_prob_analytic_fast(ps[i:max(i+cutoff,G)],lamb)
    print "computing leftward"
    print "leftward tuple:",max(i-cutoff,0),i
    print "leftward ps:",len(ps[max(i-cutoff,0):i:-1])
    leftward_prob = rightward_prob_analytic_fast(ps[min(i-cutoff,0):i:-1],lamb)
    return 1 - (1-rightward_prob)*(1-leftward_prob)
    
def point_probs(ps,mean_frag_length):
    lamb = 1.0/mean_frag_length
    G = len(ps)
    out_probs = [0] * len(ps)
    
def log_likelihood(reads,ps):
    return sum((log(read_likelihood(read,ps))) for read in reads)

