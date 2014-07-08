import numpy as np
from math import floor
from scipy import sparse
import itertools
from project_utils import falling_fac,esp
from utils import rslice

def transfer_matrix(ki,q,sparse=False):
    Wi = np.zeros((2*q+1,2*q+1))
    for i in xrange(2*q + 1):
        row = [1,(q-(i+1)/2)*ki]
        for j in xrange(2):
            col = j + i + (i%2)
            if col <= 2*q:
                Wi[i,col] = row[j]
    return Wi
            
def scan_transfer_matrices(ks,q,normalize=True):
    """Scan an array of sites given by binding rates ks, assuming copy
    number q.  Return an array containing weights corresponding to states:
    Uq, B(q-1),U(q-1),...,B0,U0"""
    v0 = np.array([1]+[0]*(2*q),dtype=np.float64)
    # v0 = np.zeros(2*q+1,dtype=np.float64)
    # v0[0] = 1
    for k in ks:
        v0 = v0.dot(transfer_matrix(k,q))
        if normalize:
            v0 = v0/np.linalg.norm(v0,1)
    return v0

def scan_transfer_matrices_fast(ks,q):
    #vcur = np.array([1]+[0]*(2*q),dtype=np.float64)
    vcur = [1]+[0]*(2*q)
    n = 2*q + 1
    for k in ks:
        vnext = [0]*n
        for i in xrange(2*q + 1):
            row = [1,(q-(i+1)/2)*k]
            for j in xrange(2):
                col = j + i + (i%2)
                if col <= 2*q:
                    vnext[col] += row[j]*vcur[i]
        vcur = vnext
    return vcur

        
def partition_from_scan(scan):
    """compute partition function from a scan array"""
    return sum(scan)

def partition_check(ks,q):
    """Compute partition function explicity, as a check"""
    print "G,q:",len(ks),q
    return sum(falling_fac(q,j)*esp(ks,j) for j in verbose_gen(range(q+1)))

def occupancies_check(ks,q):
    Z = partition_check(ks,q)
    def Zi(i):
        ki = ks[i]
        new_ks = ks[:]
        new_ks.remove(ki)
        return q*ki*partition_check(new_ks,q-1)
    return [Zi(i)/float(Z) for i in range(len(ks))]

def occupancy_check(ks,q):
    """Compute occupancy explicity, as a check"""
    return sum(j*falling_fac(q,j)*esp(ks,j) for j in range(q+1))/float(partition_check(ks,q))

def mean_occupancy_from_scan(scan):
    """Compute mean number of TFs bound to chromosome"""
    n = len(scan)
    assert n % 2 == 1
    q = (n-1)/2
    #free_vector = np.array([q-(i+1)/2 for i in range(n)])
    occ_vector = np.array([(i+1)/2 for i in range(n)])
    return scan.dot(occ_vector)/partition_from_scan(scan)

def occupancy_probs_from_scan(scan):
    """Compute probability of each chromosomal occupancy level from scan"""
    n = len(scan)
    assert n % 2 == 1
    q = (n-1)/2
    #free_vector = np.array([q-(i+1)/2 for i in range(n)])
    occ_vector = [(i+1)/2 for i in range(n)]
    Z = partition_from_scan(scan)
    def scan_at_occ(j):
        """Filter scan for levels having chromosomal occupancy j"""
        return [scan[i] for i in range(len(scan)) if occ_vector[i]==j]
    return [sum(scan_at_occ(occ))/Z
            for occ in range(q+1)]

def occupancies2(ks,q):
    """Wrong"""
    scan = scan_transfer_matrices(ks,q)
    occ_probs = occupancy_probs_from_scan(scan)
    occs = [sum(((q-occ)*k/float(1+(q-occ)*k))*occ_prob for occ,occ_prob in enumerate(occ_probs)) for k in ks]
    return occs
    
def occupancies(ks,q):
    """wrong"""
    scan = scan_transfer_matrices(ks,q,normalize=False)
    occ = mean_occupancy_from_scan(scan)
    print "occ:",occ
    fp = q - occ
    print "fp:",fp
    occs = [(fp*k/(1+fp*k)) for k in ks]
    print "total protein:",sum(occs) + fp
    return occs
    
