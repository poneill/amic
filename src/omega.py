import sympy
from itertools import combinations
from utils import product,fac
from project_utils import esp

def make_k(i):
    return sympy.var('k%s' % i)
    
def omega(N,q):
    assert 0 <= q <= N
    ks = [make_k(i) for i in range(1,N+1)]
    return sum(int(fac(q)/fac(q-i))*esp(ks,i) for i in range(q+1))

def step_n(om,nplus1,q):
    ks = [make_k(i) for i in range(1,nplus1 + 1)]
    k = make_k(nplus1)
    return om*(1+k) - q*esp(ks,nplus1)

def mat_permanent(M):
    m = M.rows
    n = M.cols
    return sum(product(M[i,perm[i]] for i in range(m))
               for perm in itertools.permutations(range(n)))

def make_matrix(n):
    ks = [make_k(i) for i in range(1,n + 1)]
    mat_list = [[1]*(n+1) for i in range(n+1)]
    for i in range(n):
        mat_list[i][i] = ks[i]
    return sympy.Matrix(mat_list)
