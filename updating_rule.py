from __future__ import division
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
        if v == 'Gov':
            eSum = 1
        else:
            eSum = 0
        

        for u in net.predecessors_iter(v):
            if u == 'Gov':
                prevState['Gov'] = 1
            else:
                w_uv = 0.5*net[u][v]['weight']
                eSum += w_uv * prevState[u]

        #### determine the current state for v as a function of eSum and threshold of v ####
        if eSum == 1 and prevState[v] == 1:
            currState[v] = 1
        else: 
            currState[v] = 0

        currState['Gov'] = 1

        
        if prevState['FirmA'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))

        elif prevState['FirmB'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmC'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmD'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmE'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmF'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmG'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmH'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmI'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
         
        elif prevState['FirmJ'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmK'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmL'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmM'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        
        elif prevState['FirmN'] == 1:
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN']))
        else:
            currState['Gov'] = 0

        
        
        if prevState['FirmA'] == 0:
            currState['FirmA'] = prevState['FirmA'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmB'] == 0:
            currState['FirmB'] = prevState['FirmB'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmC'] == 0:
            currState['FirmC'] = prevState['FirmC'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmD'] == 0:
            currState['FirmD'] = prevState['FirmD'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmE'] == 0:
            currState['FirmE'] = prevState['FirmE'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmF'] == 0:
            currState['FirmF'] = prevState['FirmF'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmG'] == 0:
            currState['FirmG'] = prevState['FirmG'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmH'] == 0:
            currState['FirmH'] = prevState['FirmH'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmI'] == 0:
            currState['FirmI'] = prevState['FirmI'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmJ'] == 0:
            currState['FirmJ'] = prevState['FirmJ'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmK'] == 0:
            currState['FirmK'] = prevState['FirmK'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmL'] == 0:
            currState['FirmL'] = prevState['FirmL'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmM'] == 0:
            currState['FirmM'] = prevState['FirmM'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmN'] == 0:
            currState['FirmN'] = prevState['FirmN'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        else:
            prevState['Gov'] = (prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])

        
        
        if prevState['FirmA'] == .25:
            currState['FirmA'] = prevState['FirmA'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmB'] == .25:
            currState['FirmB'] = prevState['FirmB'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmC'] == .25:
            currState['FirmC'] = prevState['FirmC'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmD'] == .25:
            currState['FirmD'] = prevState['FirmD'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmE'] == .25:
            currState['FirmE'] = prevState['FirmE'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmF'] == .25:
            currState['FirmF'] = prevState['FirmF'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmG'] == .25:
            currState['FirmG'] = prevState['FirmG'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmH'] == .25:
            currState['FirmH'] = prevState['FirmH'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmI'] == .25:
            currState['FirmI'] = prevState['FirmI'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmJ'] == .25:
            currState['FirmJ'] = prevState['FirmJ'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmK'] == .25:
            currState['FirmK'] = prevState['FirmK'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmL'] == .25:
            currState['FirmL'] = prevState['FirmL'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmM'] == .25:
            currState['FirmM'] = prevState['FirmM'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmN'] == .25:
            currState['FirmN'] = prevState['FirmN'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        else:
            prevState['Gov'] = (prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])


        if prevState['FirmA'] == .5:
            currState['FirmA'] = prevState['FirmA'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmB'] == .5:
            currState['FirmB'] = prevState['FirmB'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmC'] == .5:
            currState['FirmC'] = prevState['FirmC'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmD'] == .5:
            currState['FirmD'] = prevState['FirmD'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmE'] == .5:
            currState['FirmE'] = prevState['FirmE'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmF'] == .5:
            currState['FirmF'] = prevState['FirmF'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmG'] == .5:
            currState['FirmG'] = prevState['FirmG'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmH'] == .5:
            currState['FirmH'] = prevState['FirmH'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmI'] == .5:
            currState['FirmI'] = prevState['FirmI'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmJ'] == .5:
            currState['FirmJ'] = prevState['FirmJ'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
       
        if prevState['FirmK'] == .5:
            currState['FirmK'] = prevState['FirmK'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmL'] == .5:
            currState['FirmL'] = prevState['FirmL'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmM'] == .5:
            currState['FirmM'] = prevState['FirmM'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        
        if prevState['FirmN'] == .5:
            currState['FirmN'] = prevState['FirmN'] + .25
            currState['Gov'] = ((prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])) - .25
        else:
            prevState['Gov'] = (prevState['FirmA']) + (prevState['FirmB']) + (prevState['FirmC']) + (prevState['FirmD']) + (prevState['FirmE']) + (prevState['FirmF']) + (prevState['FirmG']) + (prevState['FirmH']) + (prevState['FirmI']) + (prevState['FirmJ']) + (prevState['FirmK']) + (prevState['FirmL']) + (prevState['FirmM']) + (prevState['FirmN'])

        if currState['Gov'] > 10.5:
            currState['Gov'] = 4  
        elif currState['Gov'] > 7:
            currState['Gov'] = 3
        elif currState['Gov'] > 3.5:
            currState['Gov'] = 2
        elif currState['Gov'] > 0:
            currState['Gov'] = 1 
    return currState



    
################# end: sigmoid_updating ########################

def main():
    print "updating_rule module is the main code."
    EDGE_FILE = 'C:\Boolean_Delay_in_Economics\Gov\EDGE_FILE.dat'
    NODE_FILE = 'C:\Boolean_Delay_in_Economics\Gov\NODE_FILE.dat'
    
    net = inet.read_network_from_file(EDGE_FILE, NODE_FILE)

    #prevState = {'a':0.0, 'b':0.0, 'c':1.0}
    prevState= {'FirmA': 0, 'FirmB': 1, 'FirmC': 1, 'FirmD': 1, 'FirmE': 1, 'FirmF': 1, 'FirmG': 1, 'FirmH': 1,'FirmI': 1, 'FirmJ': 1, 'FirmK': 1, 'FirmL': 1, 'FirmM': 1, 'FirmN': 1}
 
    print "network state @ previous step", OrderedDict(sorted(prevState.items(), key=lambda t: t[0]))
    
    currState = sigmoid_updating(net, prevState)
    print "network state @ current step", OrderedDict(sorted(currState.items(), key=lambda t: t[0]))

if __name__=='__main__':
    main()