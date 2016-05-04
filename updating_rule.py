__author__ = '''Hyunju Kim'''


import os
import sys
import numpy as np
import networkx as nx
from collections import OrderedDict

import input_net as inet

################# begin: sigmoid_updating ######################
def sigmoid_updating(net, prevState):
    """Update according to fixed thresholds for each node"""

    '''
        Arguments:
                1. net
                2. prevState
        Return:
               1. currState
    '''
    
    currState = {}
    #### compute the current states of nodes in the net ####
    for v in net.nodes():
                #### compute weighted sum for node v over its neighbors u ####
        eSum = 0
        for u in net.predecessors_iter(v):
            w_uv = 0.5*net[u][v]['weight']
            eSum += w_uv * prevState[u]
        #### determine the current state for v as a function of eSum and threshold of v ####
        if eSum == 1 or prevState[v] == 1:
            currState[v] = 1
        else: 
            currState[v] = 0


    return currState


    
################# end: sigmoid_updating ########################

def main():
    print "updating_rule module is the main code."
    EDGE_FILE = 'C:\Boolean_Delay_in_Economics\Manny\EDGE_FILE.dat'
    NODE_FILE = 'C:\Boolean_Delay_in_Economics\Manny\NODE_FILE.dat'
    
    net = inet.read_network_from_file(EDGE_FILE, NODE_FILE)

    #prevState = {'a':0.0, 'b':0.0, 'c':1.0}
    prevState= {'FirmA': 0, 'FirmB': 1, 'FirmC': 1, 'FirmD': 1, 'FirmE': 1, 'FirmF': 1, 'FirmG': 1, 'FirmH': 1,'FirmI': 1, 'FirmJ': 1, 'FirmK': 1, 'FirmL': 1, 'FirmM': 1, 'FirmN': 1}
 
    print "network state @ previous step", OrderedDict(sorted(prevState.items(), key=lambda t: t[0]))
    
    currState = sigmoid_updating(net, prevState)
    print "network state @ current step", OrderedDict(sorted(currState.items(), key=lambda t: t[0]))

if __name__=='__main__':
    main()