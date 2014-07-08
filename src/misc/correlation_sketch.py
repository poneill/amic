"""
Does independence sampling reduce correlations in MCMC?  A simple sketch...
"""

from utils import normalize,mh
import random

ps = normalize(range(1,100+1))

def P(x):
    return ps[x-1]

def Q1(x):
    r = random.random()
    if r < 1/3.0:
        direction = -1
    elif r < 2/3.0:
        direction = 0
    else:
        direction = 1
    if (x == 1 and direction == -1) or (x == 100 and direction == 1):
        direction = 0
    return x + direction

def dQ1(xp,x):
    if 1<x<101 or xp != x:
        return 1/3.0
    else:
        return 2/3.0
    
def Q2(x):
    return random.choice(range(1,100+1))

def main():
    chain1 = mh(P,Q1,1,dprop=dQ1)
    chain2 = mh(P,Q2,1)
        
        
