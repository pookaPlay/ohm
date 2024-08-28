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
        
        self.networkStats = dict()
        self.networkStats['minWeightIncrease'] = list()
        self.networkStats['maxWeightIncrease'] = list()

    
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
        #print(f"Analyzing {numLayers} layers")        
        for li in range(numLayers): 
            self.localResults[li] = self.ohm.stackMem[li].GetLSBInts()
            self.AnalyzeList(li, self.localResults[li])
        
        self.networkStats['minWeightIncrease'].append(self.ohm.minWeightIncrease)                
        self.networkStats['maxWeightIncrease'].append(self.ohm.maxWeightIncrease)

    def PrintSomeStats(self):        
        print(f"Weight updates MIN: {self.networkStats['minWeightIncrease']} MAX: {self.networkStats['maxWeightIncrease']}")

    def PlotByLayer(self):
   
        #######################################################################
        #######################################################################
        ## Value plots
        statPlot = False
        if statPlot:
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
        sumFlag = np.zeros_like(ticksTaken)
        weightNetwork = np.zeros(())

        for li in range(len(ticksTaken)):
            for ni in range(len(ticksTaken[li])):
                valNetwork[li][ni] = self.localResults[li][ni]
                ticksTaken[li][ni] = ticksTaken[li][ni] #/ K                

        # Transpose the arrays to swap x and y axes
        ticksTaken = ticksTaken.T
        valNetwork = valNetwork.T

        fig = plt.figure(1)        
        ax3 = fig.add_subplot(231)
        ax4 = fig.add_subplot(232)
        ax1 = fig.add_subplot(233)
        ax2 = fig.add_subplot(236)
        ax5 = fig.add_subplot(234)
        ax6 = fig.add_subplot(235)
        
        # 2D Heatmaps
        cax3 = ax3.imshow(ticksTaken, cmap='viridis', aspect='auto')
        cax4 = ax4.imshow(valNetwork, cmap='viridis', aspect='auto')
        fig.colorbar(cax3, ax=ax3, orientation='vertical')
        fig.colorbar(cax4, ax=ax4, orientation='vertical')
        ax3.set_ylabel('Input')
        ax3.set_xlabel('Layer')
        ax3.set_title('Ticks')
        ax4.set_ylabel('Input')
        ax4.set_xlabel('Layer')
        ax4.set_title('Values')
        
        allweights, allthresh = self.ohm.GetPrettyParameters()

        cax1 = ax1.imshow(allweights, cmap='viridis', aspect='auto')
        cax2 = ax2.imshow(allthresh, cmap='viridis', aspect='auto')
        fig.colorbar(cax1, ax=ax1, orientation='vertical')
        fig.colorbar(cax2, ax=ax2, orientation='vertical')
        ax1.set_xlabel('Weights')
        ax1.set_ylabel('Layer')
        ax1.set_title('Weights')
        ax2.set_xlabel('Thresh')
        ax2.set_ylabel('Layer')
        ax2.set_title('Thresh')
        
        sumFlag = np.array(self.ohm.sumOut)
        sumFlag = sumFlag.T

        ax5.set_ylabel('Input')
        ax5.set_xlabel('Layer')
        ax5.set_title('SumFlag')
        cax5 = ax5.imshow(sumFlag, cmap='viridis', aspect='auto')        
        fig.colorbar(cax5, ax=ax5, orientation='vertical')

        thresh = int(self.param['numInputs'])    
        #print(f"Half D: {thresh}")
        sumFlag2 = (sumFlag == thresh)

        cax6 = ax6.imshow(sumFlag2, cmap='viridis', aspect='auto')
        fig.colorbar(cax6, ax=ax6, orientation='vertical')
        ax6.set_ylabel('Input')
        ax6.set_xlabel('Layer')
        ax6.set_title('SumFlag == halfD')

        plt.tight_layout()
        plt.show()
        
    
        #ax1 = fig.add_subplot(221, projection='3d')
        #X, Y = np.meshgrid(range(ticksTaken.shape[1]), range(ticksTaken.shape[0]))
        #ax.plot_surface(X, Y, ticksTaken, cmap='viridis')
        #ax.view_init(elev=90, azim=0)

    