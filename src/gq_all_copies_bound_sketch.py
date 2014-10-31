from itertools import combinations

G = 5
q = 3
states = list(combinations(range(G),q))
eps = range(G)

def weight(state):
    return exp(-sum(state))

def prob_ref(i):
    states_i = i_states(i)
    Z_i = sum(weight(state) for state in states_i)
    Z = sum(weight(state) for state in states)
    return Z_i/Z

def i_states(i):
    return [state for state in states if i in state]

def non_i_states(i):
    return [state for state in states if not i in state]
    
def prob_spec(i):
    # Z_i = 1
    # Z = 1 + sum(map(weight,non_i_states(i)))/sum(map(weight,i_states(i)))
    # return Z_i/Z
    return 1/(1+exp(eps[i])*(sum(exp(-eps[j]) for j in range(G) if not j == i))/q)

def to_bin(state):
    xs = [0]*G
    for s in state:
        xs[s] = 1
    return "".join(map(str,xs))
    
    
