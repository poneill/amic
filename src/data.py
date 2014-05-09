"""
This file contains data for the genome (GENOME) and DNA binding domain (TRUE_ENERGY_MATRIX)
"""

W = 5 # width of DNA binding domain

TRUE_ENERGY_MATRIX = ([[-2,0,0,0] for i in range(W)]) 
with open('genome.fa') as f:
    lines = f.readlines()
GENOME = lines[1]
L = len(GENOME)
MEAN_FRAG_LENGTH = 50
