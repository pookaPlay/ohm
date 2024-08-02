from bls.OHM_ADDER_CHANNEL import OHM_ADDER_CHANNEL
from bls.STACK_BLS import STACK_BLS
from bls.BSMEM import BSMEM
import networkx as nx
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def count_monotonic_pairs(lst):
    increasing_pairs = 0
    decreasing_pairs = 0
    equal_pairs = 0

    for x, y in zip(lst, lst[1:]):
        if x < y:
            increasing_pairs += 1
        elif x > y:
            decreasing_pairs += 1
        elif x == y:
            equal_pairs += 1
    
    total_pairs = len(lst) - 1
    if total_pairs == 0:
        return 0, 0  # Avoid division by zero for lists with fewer than 2 elements
    
    normalized_increasing = increasing_pairs / total_pairs
    normalized_decreasing = decreasing_pairs / total_pairs
    normalized_equal = equal_pairs / total_pairs

    return normalized_increasing, normalized_decreasing, normalized_equal


class OHM_PROBE:

    def __init__(self, param, ohm):
    
        self.param = param              
        self.ohm = ohm

        self.featuresByLayer = ['mean', 'variance', 'incPairs', 'decPairs', 'eqPairs']

        self.statsByLayer = dict()
        for f in self.featuresByLayer:
            self.statsByLayer[f] = dict()

    
    def AnalyzeList(self, key, results):

        mean = sum(results) / len(results)
        variance = sum((x - mean) ** 2 for x in results) / len(results)
        self.statsByLayer['mean'][key] = mean
        self.statsByLayer['variance'][key] = variance

        incPairs, decPairs, eqPairs = count_monotonic_pairs(results)
        self.statsByLayer['incPairs'][key] = incPairs
        self.statsByLayer['decPairs'][key] = decPairs
        self.statsByLayer['eqPairs'][key] = eqPairs

    def AnalyzeRun(self):
        numLayers = len(self.ohm.stackMem)
        self.localResults = dict()
        print(f"Analyzing {numLayers} layers")        
        for li in range(numLayers): 
            self.localResults[li] = self.ohm.stackMem[li].GetLSBInts()
            self.AnalyzeList(li, self.localResults[li])

        print(f"Plot Results")
        self.PlotByLayer()                

    def PlotByLayer(self):
   
        #######################################################################
        #######################################################################
        ## Value plots
        fig, axes = plt.subplots(2, 1)  # Create a grid of 2 subplots

        plt1 = ['mean', 'variance']
        plt2 = ['incPairs', 'decPairs', 'eqPairs']

        for f in plt1:
            axes[0].plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        axes[0].legend()

        for f in plt2:
            axes[1].plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        axes[1].legend()
        
        
        #######################################################################        
        #######################################################################
        ## Surface plots
        K = self.param['memK']
        ticksTaken = self.ohm.doneOut
        assert(len(ticksTaken) == len(self.localResults))
        
        # Convert ticksTaken to a numpy array for easier manipulation
        ticksTaken = np.array(ticksTaken)
        valNetwork = np.zeros_like(ticksTaken)
        for li in range(len(ticksTaken)):
            for ni in range(len(ticksTaken[li])):
                valNetwork[li][ni] = self.localResults[li][ni]
                ticksTaken[li][ni] = ticksTaken[li][ni] / K

        fig = plt.figure()
        fig2 = plt.figure()
        # Create a meshgrid for X and Y coordinates
        X, Y = np.meshgrid(range(ticksTaken.shape[1]), range(ticksTaken.shape[0]))                
        ax = fig.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, ticksTaken, cmap='viridis')
        ax2.plot_surface(X, Y, valNetwork, cmap='viridis')
        ax.view_init(elev=90, azim=0)
        ax2.view_init(elev=90, azim=0)
        ax.set_xlabel('Input')
        ax.set_ylabel('Layer')
        ax.set_zlabel('Ticks Taken')
        ax2.set_xlabel('Input')
        ax2.set_ylabel('Layer')
        ax2.set_zlabel('Values')


        plt.show()
    

    