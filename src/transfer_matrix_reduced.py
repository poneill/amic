"""
See transfer_matrix.py.  We wish to reduce the dimension of the transfer matrices from 2q+1 to q.
"""
import numpy as np
from numpy import linalg
import sympy
from sympy.matrices import *
from project_utils import falling_fac,esp
from scipy.sparse.linalg import lgmres
from utils import show,choose,fac
from math import exp

beta = 1.0

def lgmres_solve(A,b):
    """Solve Ax = b w/ lgmres"""
    x, info = lgmres(A,b,maxiter=10000)
    if info != 0:
        raise Exception("lgmres failed on A=%s,b=%s, info:%s" % (A,b,info))
    return x

def test_pd(M,trials=100):
    """Test whether M is positive definite"""
    n = len(M)
    for i in xrange(trials):
        v = np.array([random.random()-0.5 for j in range(n)])
        if v.dot(M).dot(v) < 0:
            print "FAILED"
            return
    print "PASSED with %s trials" % trials
    
def transfer_matrix_r(ki,q):
    Wi = np.eye(q+1,dtype=np.float64)
    for j in range(q):
        Wi[(j,j+1)] = (q-j)*ki
    return Wi

def log_transfer_matrix_r(ki,q):
    def a(n,k):
        """See OEIS A111492"""
        return fac(k-1)*choose(n,k)
    A = [[(-1)**(j+1+i)*a(q-i,j-i)*ki**(j-i) if j > i else 0
          for j in range(q+1)] for i in range(q+1)]
    return np.matrix(A)

def transfer_matrix_prob(ki,q):
    Wi = np.eye(q+1)
    for j in range(q):
        z = float(1 + (q-j)*ki)
        Wi[(j,j)] = 1/z
        Wi[(j,j+1)] = (q-j)*ki/z
    return Wi


def transfer_matrix_dki(q):
    Wi = np.zeros((q+1,q+1))
    for j in range(q):
        Wi[(j,j+1)] = (q-j)
    return Wi

def log_transfer_matrix_dki(ki,q):
    def a(n,k):
        """See OEIS A111492"""
        return fac(k-1)*choose(n,k)
    A = [[(-1)**(j+1+i)*a(q-i,j-i)*(j-i)*ki**(j-i-1) if j > i else 0
          for j in range(q+1)] for i in range(q+1)]
    return np.matrix(A)

def test_log_transfer_matrix_dki(ki,q):
    M = transfer_matrix(7,4)
    dM = 0 
    logM = logm(M)
    pass
    
def transfer_matrix_dki_prob(k,q):
    Wi = np.zeros((q+1,q+1))
    for j in range(q):
        Qeff = q-j
        Wi[j,j] = -Qeff/float(1+Qeff*k)
        Wi[j,j+1] = Qeff/float(1+Qeff*k)
    return Wi

    
def transfer_matrix_sym(ki,q):
    def f(i,j):
        if i == j:
            return 1
        elif j == i+1:
            return (q-j+1)*ki
        else:
            return 0
    return Matrix(q+1,q+1,f)
    
def transfer_matrix_inv(ki,q):
    def f(i,j):
        if i == j:
            return 1
        elif j > i:
            return ki**(j-i)*falling_fac(q-i,j-i)*(-1)**j*(-1)**i
        else:
            return 0
    mat = [[f(i,j) for j in range(q+1)]
           for i in range(q+1)]
    return np.matrix(mat)

def transfer_matrix_inv_stable(ki,q,v):
    """Compute v^T * Wi^-1 without computing Wi^-1 explicitly"""
    # v^T = V0^T*Wi
    v0 = np.array(v)
    for i in range(1,q+1):
        v0[i] = v[i]-(q-i+1)*ki*v0[i-1]
    v_recovered = v0.dot(transfer_matrix_r(ki,q))
    print v_recovered == v
    return v0

def test_transfer_matrix_inv_stable():
    ks = range(1,21)
    q = 23
    k = 20
    scan = scan_transfer_matrices_r(ks,q)
    scan_implicit_inversion = transfer_matrix_inv_stable(k,q,scan)
    scan_explicit_inversion = scan.dot(transfer_matrix_inv(k,q))
    new_ks = ks[:]
    new_ks.remove(k)
    scan_paranoid_inversion = scan_transfer_matrices_r(new_ks,q)
    scan_implicit_recovered = scan_implicit_inversion.dot(transfer_matrix_r(k,q))
    scan_explicit_recovered = scan_explicit_inversion.dot(transfer_matrix_r(k,q))
    scan_paranoid_recovered = scan_paranoid_inversion.dot(transfer_matrix_r(k,q))
    print scan_implicit_inversion == scan_explicit_inversion
    
def troubleshooting():
    ks,q = find_mwe()
    k = 1
    # motivation:
    problem_occs = occupancies(ks,q)
    # occupancies exceed 1
    # ---
    scan = scan_transfer_matrices_r(ks,q) # scan is correct
    vf = np.array([1]*(q+1)) # correct
    wJ = transfer_matrix_inv_stable(k,q,scan) # cannot be correct: negative entries
    v1_inv_correct = [0.05] + [0]*q
    v1_inv_explicit = v1.dot(transfer_matrix_inv(k,19))
    v1_inv_implicit = transfer_matrix_inv_stable(k,q,scan)

def find_mwe():
    """Find the lowest values of ks, q such that occupancies exceed 1"""
    k = 2
    q = 0
    found_yet = False
    while not found_yet:
        print k,q
        ks = range(1,k)
        occs = occupancies(ks,q)
        if max(occs) > 1:
            found_yet = True
        else:
            q += 1
    return ks,q
    
def scan_transfer_matrices_r(ks,q):
    v0 = np.array([1] + [0]*q)
    for k in ks:
        v0 = v0.dot(transfer_matrix_r(k,q))
    return v0

def scan_transfer_matrices_r_experimental(ks,q):
    v0 = np.array([1] + [0]*q)
    total_norm = 1
    for k in ks:
        v0 = v0.dot(transfer_matrix_r(k,q))
        cur_norm = linalg.norm(v0)
        total_norm *= cur_norm
        v0 = v0/cur_norm
    return total_norm * v0

def scan_transfer_matrices_prob(ks,q):
    v0 = np.array([1] + [0]*q)
    for k in ks:
        v0 = v0.dot(transfer_matrix_prob(k,q))
    return v0

def occupancies_ref(ks,q):
    def remove(xs,x):
        xs_new = xs[:]
        xs_new.remove(x)
        return xs_new
    Z = float(sum(falling_fac(q,n)*esp(ks,n) for n in range(q+1)))
    print "Z:",Z
    return [(q*k*sum(show(falling_fac(q-1,n)*esp(remove(ks,k),n)) for n in range(q)))/Z for k in ks]

    
def occupancies_ref2(ks,q,scan=None):
    if scan is None:
        scan = scan_transfer_matrices_r(ks,q)
    vf = np.array([1]*(q+1))
    Z = scan.dot(vf)
    return [(scan.dot(transfer_matrix_inv(k,q)).dot(transfer_matrix_dki(q)).dot(vf)*k)/Z for k in ks]

def occupancies_norm(ks,q,scan=None):
    if scan is None:
        scan = scan_transfer_matrices_r_experimental(ks,q)
    vf = np.array([1]*(q+1))
    Z = scan.dot(vf)
    return [(scan.dot(transfer_matrix_inv(k,q)).dot(transfer_matrix_dki(q)).dot(vf)*k)/Z for k in ks]

def occupancies(ks,q,scan=None):
    if scan is None:
        scan = scan_transfer_matrices_r(ks,q)
    vf = np.array([1]*(q+1))
    Z = scan.dot(vf)
    return [lgmres_solve(np.transpose(transfer_matrix_r(k,q)),scan).dot(transfer_matrix_dki(q)).dot(vf)*k/Z
            for k in ks]


def occupancies_prob(ks,q,scan=None):
    if scan is None:
        scan = scan_transfer_matrices_prob(ks,q)
    vf = np.array([1]*(q+1))
    Z = scan.dot(vf)
    return [lgmres_solve(np.transpose(transfer_matrix_prob(k,q)),scan).dot(transfer_matrix_dki_prob(k,q)).dot(vf)*k/Z
            for k in ks]

def normalize_array(arr):
    return arr/np.linalg.norm(arr,1)

def expm(A,n=100):
    return sum(linalg.matrix_power(A,i)/fac(i) for i in range(n+1))
    
def logm(A,n=100):
    I = np.eye(len(A))
    X = A - I
    return sum(linalg.matrix_power(X,i)/float(i)*(-1)**(i+1) for i in range(1,n+1))

def log_property_test():
    """Verify that log(prod W) = sum(log(W))"""
    ks = [1,2,3,4]
    q = 3
    Zmat = expm(sum(logm(transfer_matrix_r(k,q)) for k in ks))
    Zmat_ref = reduce(np.dot,[transfer_matrix_r(k,q) for k in ks])
    return Zmat == Zmat_ref

def log_property_test2():
    """Verify that log(v0 prod W vf) = v0 sum(log(W)) vf"""
    G = 10
    ks = [exp(random.random()) for i in range(G)]
    q = 3
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    Zmat_ref = reduce(np.dot,[transfer_matrix_r(k,q) for k in ks])
    Z_ref = v0.dot(Zmat_ref).dot(vf)
    Zmat = expm(sum(logm(transfer_matrix_r(k,q)) for k in ks))
    Z = v0.dot(Zmat_ref).dot(vf)
    print Z,Z_ref
    return abs(Z - Z_ref) < 10**-10

def log_property_test3():
    """Verify that log(v0 prod W vf) = v0 sum(log(W)) vf"""
    G = 10
    ks = [exp(random.random()) for i in range(G)]
    q = 3
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    Zmat = sum(logm(transfer_matrix_r(k,q)) for k in ks)
    logZ = logmv0.dot(Zmat).dot(vf)
    Zmat_ref = reduce(np.dot,[transfer_matrix_r(k,q) for k in ks])
    logZ_ref = log(v0.dot(Zmat).dot(vf))
    print logZ,logZ_ref
    return abs(logZ - logZ_ref) < 10**-10

def Z(ks,q):
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    return v0.dot(Zmat(ks,q)).dot(vf)
    
def Zmat(ks,q):
    return reduce(lambda x,y:x.dot(y),[transfer_matrix_r(k,q) for k in ks])

def log_Zmat(ks,q):
    return reduce(lambda x,y:x + y,[log_transfer_matrix_r(k,q) for k in ks])

def log_Zmat_ref(ks,q):
    return log(Zmat(ks,q))

def log_property_test4():
    """Can we compute occupancies just by taking derivative of Wi?"""
    pass

def occupancies_speculative(ks,q):
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    return [-1/beta*v0.dot(log_transfer_matrix_dki(k,q)).dot(vf) for k in ks]


def occupancies_logtest(ks,q):
    """Compute occupancies by matrix logarithm scheme"""
    Zmat = expm(sum(log_transfer_matrix_r(k,q) for k in ks))
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    Z = v0.dot(Zmat).dot(vf)
    occs = [(v0.dot(Zmat.dot(transfer_matrix_inv(k,q)).dot(transfer_matrix_dki(q))).dot(vf)/Z)[0,0] for k in ks]
    #occs = [((v0.dot(expm(logm(Zmat)-log_transfer_matrix_r(k,q))).dot(transfer_matrix_dki(q))).dot(vf)/Z)[0,0] for k in ks]
    return occs

def occupancies_logtest2(ks,q):
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    
def occ_logtest_troubleshooting():
    q = 20
    ks = [1,2,3,4]
    Zmat = expm(sum(logm(transfer_matrix_r(k,q)) for k in ks))
    v0 = np.array([1] + [0]*(q))
    vf = np.array([1]*(q+1))
    my_scan = v0.dot(Zmat) # bad oscillations starting at about my_scan[10-12]
    ref_scan = scan_transfer_matrices_r(ks,q)
    Zmat_ref = reduce(lambda x,y:x.dot(y),[transfer_matrix_r(k,q) for k in ks])
