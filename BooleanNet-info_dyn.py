#!/usr/bin/python

#BooleanNet-InfoDyn.py

#bioinfo.py


__author__ = '''Hyunju Kim'''

import sys
import os
import random as ran
from math import log
from optparse import OptionParser, OptionGroup
from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import defaultdict
import operator
import draw_plots
from collections import OrderedDict

import input_net as inet
import updating_rule as ur
import time_evol as tev
import info_dyn as info


def main(args):

    maxStep = 50 # the length of time steps starting from each initail state
    Nbr_States = 2 # the number of all possible states of a node
    historyLength = 1 # the length of history states
    
    ## 1. Network Information from Data File ##
    EDGE_FILE = 'C:\Boolean_Delay_in_Economics\Manny\EDGE_FILE.dat'
    NODE_FILE = 'C:\Boolean_Delay_in_Economics\Manny\NODE_FILE.dat'
    
    #EDGE_FILE = '../data/budding-net/budding-net-edges.txt'
    #NODE_FILE = '../data/budding-net/budding-net-nodes.txt'
    
    ## 2. To Build net and nodes_list: Module 'input_net' is required ##
    net = inet.read_network_from_file(EDGE_FILE, NODE_FILE)
    nodes_list = inet.build_nodes_list(NODE_FILE)

    ## 5. To generate time series data from 1, 2 and 3: Modules 'time_evol' and 'updating_rule' are required ##
    timeSeries_Type = ['one_trajectory'] # 'all_initial', 'primary_attractor', 'one_trajectory'

    if 'all_initial' in timeSeries_Type:
        ## 5-1. time series for all possible initial network states.
        Nbr_All_Initial_States = np.power(Nbr_States, len(nodes_list)) # the number of initial states of the network
        timeSeriesAll = tev.time_series_all(net, nodes_list, Nbr_All_Initial_States, Nbr_States, maxStep) # To generate time series data over all possible initial states
    
    if 'primary_attractor' in timeSeries_Type:
        ## 5-2. time series for initial network states that converge to primary (or biological) attractor
        decStateTransMap = tev.net_state_transition(net, nodes_list, Nbr_States) # To build transition map between network states over network state space
        attractors = tev.find_attractor(decStateTransMap) # To find attractors ==> attractors = {attractor state1: {'type': 'fixed', 'basin-size': 2, 'basin': [network state a, network state b]}, attractor state2: {},  ... }
        primary_attractor = attractors.keys()[0] #attractors is ordered with basin size from the function "find_attractor" and hence primary attractor is the first one.
        Initial_States_List = list(attractors[primary_attractor]['basin']) # To assign all initial states in the basin of primary attractor to the list of initail states
        timeSeriesPa = tev.time_series_pa(net, nodes_list, Initial_States_List, Nbr_States, maxStep) # To generate time series data over all initial states that converge to the primary attractor
    
    if 'one_trajectory' in timeSeries_Type:
        ## 5-3. time series for one initial network state
        InitBiState = { 'FirmA': 0, 'FirmB': 1, 'FirmC': 1, 'FirmD': 1, 'FirmE': 1, 'FirmF': 1, 'FirmG': 1, 'FirmH': 0,'FirmI': 1, 'FirmJ': 1, 'FirmK': 1, 'FirmL': 1, 'FirmM': 1, 'FirmN': 1 } # initial state for fission yeast
        #InitBiState = {'Cln3': 1, 'MBF':0, 'SBF':  0, 'Cln1_2':0, 'Cdh1':1, 'Swi5':0, 'Cdc20_Cdc14':0, 'Clb5_6':0, 'Sic1': 1, 'Clb1_2':0, 'Mcm1_SFF':0} # initial state for budding yeast
        timeSeriesOne = tev.time_series_one(net, nodes_list, InitBiState, Nbr_States, maxStep)  # To generate time series data for one particular initial state


    '''
    The format for timeSeries
    ==>
        timeSeries = { node1: {1st initial_state: [0,1,1, ......, 0], ...... , Nth initial_state: [1,0,1, ......, 1]},
                               node 2: {1st initial_state: [1,1,1, ......, 0], ...... , Nth initial_state: [1,0,0, ......, 0]}, .......}
    ==> 
        timeSeries[node2][initial_state 1] = [1,1,1, ......, 0]
    '''

    ## 4. Output files ##
    #result_ai = open('../results/fission-net/ai-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
    #result_te = open('../results/fission-net/te-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
   
    

    ## 6. To measure active information (AI) and transfer entropy (TE): using 'compute_AI' and 'compute_TE' from Module 'info_dyn" ##

    ## 6-1. For all possible initial network states
    if 'all_initial' in timeSeries_Type:
        
        ## 6-1-a. To compute AI
        result_ai_all = open('C:/Boolean_Delay_in_Economics/Manny/Results/ai-all-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
        AI_all = {}
        for n in nodes_list:
            AI_all[n] = info.compute_AI(timeSeriesAll[n], historyLength, Nbr_All_Initial_States, Nbr_States)
            result_ai_all.write('%s\t%f\n'%(n, AI_all[n]))
            print n, AI_all[n]
        
        ## 6-1-b. To compute TE
        result_te_all = open('C:/Boolean_Delay_in_Economics/Manny/Results/te-all-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
        TE_all =  defaultdict(float)
        for v in nodes_list:
            for n in nodes_list:
                TE_all[(v, n)] = info.compute_TE(timeSeriesAll[v], timeSeriesAll[n], historyLength, Nbr_All_Initial_States, Nbr_States)
                result_te_all.write('%s\t%s\t%f\n'%(v, n,TE_all[(v, n)] ))
                print v, n,TE_all[(v, n)]
    
        ## 6-1-c. Scale behavior for AI (optional)
        result_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/ai-all-scale-step%d-trans0-h%d.dat'%(maxStep, historyLength)
        viz_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/ai-all-scale-step%d-trans0-h%d.pdf'%(maxStep, historyLength)
        draw_plots.plot_AI_scale(AI_all, result_file_name, viz_file_name) ### plot and result file for AI scale

        ## 6-1-d. Scale behavior for TE (optional)
        result_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/te-all-scale-step%d-trans0-h%d.dat'%(maxStep, historyLength)
        viz_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/TransferEntropy-all-scale-step%d-trans0-h%d.pdf'%(maxStep, historyLength)
        draw_plots.plot_TE_scale(TE_all, result_file_name, viz_file_name) ### plot and result file for TE scale



    ## 6-2. For initial network states that converge to the primary attractor
    if 'primary_attractor' in timeSeries_Type:
 
        ## 6-2-a. To compute AI
        result_ai_pa = open('C:/Boolean_Delay_in_Economics/Manny/Results/ai-pa-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
        AI_pa = {}
        for n in nodes_list:
            AI_pa[n] = info.compute_AI(timeSeriesPa[n], historyLength, len(Initial_States_List), Nbr_States)
            result_ai_pa.write('%s\t%f\n'%(n, AI_pa[n]))

        ## 6-2-b. To compute TE
        result_te_pa = open('C:/Boolean_Delay_in_Economics/Manny/Results/te-pa-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
        TE_pa =  defaultdict(float)
        for v in nodes_list:
            for n in nodes_list:
                TE_pa[(v, n)] = info.compute_TE(timeSeriesPa[v], timeSeriesPa[n], historyLength, len(Initial_States_List), Nbr_States)
                result_te_pa.write('%s\t%s\t%f\n'%(v, n,TE_pa[(v, n)] ))

        ## 6-2-c. Scale behavior for AI (optional)
        result_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/ai-pa-scale-step%d-trans0-h%d.dat'%(maxStep, historyLength)
        viz_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/ActiveInformation-pa-scale-step%d-trans0-h%d.pdf'%(maxStep, historyLength)
        draw_plots.plot_AI_scale(AI_pa, result_file_name, viz_file_name) ### plot and result file for AI scale
        
        ## 6-2-d. Scale behavior for TE (optional)
        result_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results//te-pa-scale-step%d-trans0-h%d.dat'%(maxStep, historyLength)
        viz_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/TransferEntropy-pa-scale-step%d-trans0-h%d.pdf'%(maxStep, historyLength)
        draw_plots.plot_TE_scale(TE_pa, result_file_name, viz_file_name) ### plot and result file for TE scale
   
   
    ## 6-3. For a particular initial network state
    if 'one_trajectory' in timeSeries_Type:
        
        ## 6-3-a. To compute AI
        result_ai_one = open('C:/Boolean_Delay_in_Economics/Manny/Results/ai-one-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
        AI_one = {}
        for n in nodes_list:
            AI_one[n] = info.compute_AI(timeSeriesOne[n], historyLength, 1, Nbr_States)
            result_ai_one.write('%s\t%f\n'%(n, AI_one[n]))

        ## 6-3-b. To compute TE
        result_te_one = open('C:/Boolean_Delay_in_Economics/Manny/Results/te-one-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
        TE_one =  defaultdict(float)
        for v in nodes_list:
            for n in nodes_list:
                TE_one[(v, n)] = info.compute_TE(timeSeriesOne[v], timeSeriesOne[n], historyLength, 1, Nbr_States)
                result_te_one.write('%s\t%s\t%f\n'%(v, n,TE_one[(v, n)] ))

        ## 6-3-c. Scale behavior for AI (optional)
        result_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/ai-one-scale-step%d-trans0-h%d.dat'%(maxStep, historyLength)
        viz_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/2ActiveInformation-one-scale-step%d-trans0-h%d.pdf'%(maxStep, historyLength)
        draw_plots.plot_AI_scale(AI_one, result_file_name, viz_file_name) ### plot and result file for AI scale
        
        ## 6-3-d. Scale behavior for TE (optional)
        result_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/te-one-scale-step%d-trans0-h%d.dat'%(maxStep, historyLength)
        viz_file_name = 'C:/Boolean_Delay_in_Economics/Manny/Results/2TransferEntropy-one-scale-step%d-trans0-h%d.pdf'%(maxStep, historyLength)
        draw_plots.plot_TE_scale(TE_one, result_file_name, viz_file_name) ### plot and result file for TE scale

'''
    ## to obtain biological sequence for the Fission Yeast Cell-Cycle Net starting from biological inital state
    EDGE_FILE = 'C:\Boolean_Delay_in_Economics\Manny\EDGE_FILE.dat'
    NODE_FILE = 'C:\Boolean_Delay_in_Economics\Manny\NODE_FILE.dat'
    BIO_INIT_FILE = ''
    
    net = inet.read_network_from_file(EDGE_FILE, NODE_FILE)
    nodes_list = inet.build_nodes_list(NODE_FILE)


    #input_file_name1 = 'time-series/%s-step%d-trans0.dat'%(network_index, maxStep)
    #input_file1 = open( input_file_name1, 'r')
    
    Nbr_Initial_States = np.power(2,len(nodes_list))
    maxStep = 20
    Nbr_States = 2
    historyLength = 5
    
    result_ai = open('C:/Boolean_Delay_in_Economics/Manny/Results/ai-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')
    result_te = open('C:/Boolean_Delay_in_Economics/Manny/Results/te-step%d-trans0-h%d.dat'%(maxStep, historyLength),'w')

    timeSeries = tev.time_series_all(net, nodes_list, Nbr_Initial_States, Nbr_States, MAX_TimeStep=20)


    print 'AI'
    AI = {}
    for n in nodes_list:
        AI[n] = info.compute_AI(timeSeries[n], historyLength, Nbr_Initial_States, Nbr_States)
        result_ai.write('%s\t%f\n'%(n, AI[n]))
    print n, AI[n]
    print 'done AI'


    print 'TE'
    TE =  defaultdict(float)
    for v in nodes_list:
        for n in nodes_list:
            TE[(v, n)] = info.compute_TE(timeSeries[v], timeSeries[n], historyLength, Nbr_Initial_States, Nbr_States)
            result_te.write('%s\t%s\t%f\n'%(v, n,TE[(v, n)] ))
            print v, n, TE[(v, n)]
    print 'done TE'
'''


if __name__=='__main__':
    main(sys.argv)