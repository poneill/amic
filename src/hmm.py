import random
from utils import normalize,random_site,simplex_sample,pairs,concat,l2
from viterbi import *
beta = 1

def make_start_p(a01):
    return [1-a01,a01] + [0] * (L-1)

def make_trans_p(a01):
    return ([[1-a01,a01] + [0] * (L-1)] + 
            ([[int(i == j-1) for j in range(L+1)]
              for i in range(1,L)]) + 
            [[1-a01,a01] + [0] * (L-1)])


def baum_welch(obs,L):
    """Given sequence and bs length L, approximate MLE parameters for
    emission probabilities,transition rate a01 (background->site).
    TODO: non-uniform background frequencies"""
    states = range(L+1)
    a01 = random.random()
    start_p = make_start_p(a01)
    trans_p = make_trans_p(a01)
    emit_p = [simplex_sample(4) for state in states]
    hidden_states = [random.choice(states) for ob in obs]
    iterations = 0
    while True:
        # compute hidden states, given probs
        prob,hidden_states_new = viterbi(obs, states, start_p, trans_p, emit_p)
        # compute probs, given hidden states
        # first compute a01
        a01_new = estimate_a01(hidden_states_new)
        start_p_new = make_start_p(a01_new)
        trans_p_new = make_trans_p(a01_new)
        emit_p_new = estimate_emit_p(obs,hidden_states_new,states)
        if (start_p_new == start_p and
            trans_p_new == trans_p and
            emit_p_new == emit_p and
            hidden_states_new == hidden_states):
            break
        else:
            print iterations,a01,l2(start_p,start_p_new),
            print l2(concat(trans_p),concat(trans_p_new)),
            print l2((hidden_states),hidden_states_new)
            a01 = a01_new
            start_p = start_p_new
            trans_p = trans_p_new
            emit_p = emit_p_new
            hidden_states = hidden_states_new
            iterations += 1
    return start_p,trans_p,emit_p,hidden_states

def estimate_a01(hidden_states):
    n = d = 0
    for hs1,hs2 in pairs(hidden_states):
            if hs1 == 0:
                d += 1
                if hs2 == 1:
                    n += 1
    a01 = n/float(d) if d > 0 else 0
    return a01
        
def estimate_emit_p(obs,hidden_states,states):
    counts = [[0.01 for i in range(4)] for j in range(len(states))]
    for ob,hs in zip(obs,hidden_states):
        counts[hs][ob] += 1
    return [normalize(row) for row in counts]
    
def test_viterbi():
    site = [0,1,2,3,0,1,2,3]
    background = lambda n:[random.choice(range(4)) for i in range(n)]
    obs = (site + background(1000) +
           site + background(1000) +
           site)

    states = [0,1,2,3,4,5,6,7,8] # 0 is off, 1-8 are bs positions
    start_p = [0.99,0.01,0,0,0,0,0,0,0]
    trans_p = [[0.99,0.01,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0],
               [0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,0,1],
               [0.99,0.01,0,0,0,0,0,0,0],
               ]
    emit_p = [[0.25,0.25,0.25,0.25],
              [1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1],
              [1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]
              ]
    return viterbi(obs, states, start_p, trans_p, emit_p)

def test_baum_welch():
    site = [0,1,2,3,0,1,2,3]
    L = 8
    background = lambda n:[random.choice(range(4)) for i in range(n)]
    obs = concat([site + background(10) for i in range(100)])
    return baum_welch(obs,L)
    
