"""Visualize a wig file"""
from matplotlib import pyplot as plt

def read_wig(filename):
    with open(filename) as f:
        return [map(float,line.split()) for line in f.readlines()[2:]]

def main(filename):
    wig = read_wig(filename)
    plt.plot(wig)
    plt.show()
    
        
