"""
This file contains data for the genome (GENOME) and DNA binding domain (TRUE_ENERGY_MATRIX)
"""
from utils import random_site

W = 10 # width of DNA binding domain

TRUE_ENERGY_MATRIX = ([[-2,0,0,0] for i in range(W)]) 
with open('genome.fa') as f:
    lines = f.readlines()
#GENOME = lines[1]
GENOME = random_site(5000000)
L = len(GENOME)
MEAN_FRAG_LENGTH = 250

# Mon May 12 19:20:38 EDT 2014
# toy genome of 10k bases, MEAN_FRAG_LENGTH = 50 works perfectly
