from matplotlib import pyplot as plt
from project_utils import *

def infer_matrix_by_simulated_annealing(genome,binding_probabilities,w):
    """Find an energy matrix that maximizes agreement with binding probabilities"""
    def rss(matrix):
        sum_sq = sum((x-y)**2 for (x,y) in zip(binding_probabilities,
                                             theoretical_probabilities(matrix,genome)))
        #print "sum_sq:",sum_sq
        return sum_sq
    def propose(matrix):
        sd = 0.1
        new_matrix = [row[:] for row in matrix]
        i = random.randrange(w)
        j = random.randrange(4)
        new_matrix[i][j] += random.gauss(0,sd)
        return new_matrix
    inferred_matrix = anneal(rss,propose,random_energy_matrix(w),
                              verbose=True,k=0.01,tf=5*10**-5,
                              iterations=5000,stopping_crit = 10**-5)
    return inferred_matrix

def infer_matrix_by_gradient_descent(genome,binding_probabilities,w,iterations=None,verbose=True):
    """Find an energy matrix that maximizes agreement with binding probabilities"""
    def rss(matrix):
        sum_sq = sum((x-y)**2 for (x,y) in zip(binding_probabilities,
                                             theoretical_probabilities(matrix,genome)))
        #print "sum_sq:",sum_sq
        return sum_sq
    delta_weight = 0.5
    cur_matrix = random_energy_matrix(10)
    last_rss = 10**6
    cur_rss = rss(cur_matrix)
    iteration = 0
    if verbose:
        prob_rsss = []
        mat_rsss = []
    while cur_rss > 10**-6 and (iteration < iterations or iterations is None):
        gradient_matrix = [[0]*4 for i in range(w)]
        # Approximate the gradient
        for col in range(w):
            for base in range(4):
                prop_matrix = [row[:] for row in cur_matrix]
                prop_matrix[col][base] += delta_weight
                prop_rss = rss(prop_matrix)
                grad = (prop_rss - cur_rss)/delta_weight # lower is better!
                gradient_matrix[col][base] = grad
        # update the matrix via gradient descent
        for col in range(w):
            for base in range(4):
                cur_matrix[col][base] -= gradient_matrix[col][base]
        last_rss = cur_rss
        cur_rss = rss(cur_matrix)
        mat_rss = sum([(true_w - inferred_w)**2
                          for (true_w,inferred_w) in zip(concat(energy_matrix),concat(cur_matrix))])
        if cur_rss > last_rss:
            print "adjusting delta_weight"
            delta_weight /= 10
        iteration += 1
        if verbose and iteration % 10 == 0:
            prob_rsss.append(cur_rss)
            mat_rsss.append(mat_rss)
            print iteration,cur_rss,mat_rss,delta_weight
            print
            pprint(cur_matrix)
            print
    if verbose:
        return cur_matrix,prob_rsss,mat_rsss
    else:
        return cur_matrix

def test_simulate(steps = 100000):
    return simulate(energy_matrix,genome,steps)

def frequency_map(L,w,xs):
    """compute occupation frequency at each position, given a tf
    trajectory"""
    counts = Counter(xs)
    Z = float(sum(counts.values()))
    max_pos = L - w + 1
    return [counts[i]/Z for i in range(max_pos)]

def myCounter(xs):
    counts = defaultdict(int)
    for x in verbose_gen(xs,modulus=1000000):
        counts[x] += 1
    return counts

def plot_fragments(fragments,genome):
    plt.plot(normalize([len([(x,y) for (x,y) in fragments if x<=i<y]) for i in range(len(genome))]))
    plt.show()
