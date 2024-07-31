from bls.OHM_ADDER_CHANNEL import OHM_ADDER_CHANNEL
from bls.STACK_BLS import STACK_BLS
from bls.BSMEM import BSMEM
import networkx as nx
import matplotlib.pyplot as plt

def count_monotonic_pairs(lst):
    increasing_pairs = 0
    decreasing_pairs = 0
    
    for x, y in zip(lst, lst[1:]):
        if x < y:
            increasing_pairs += 1
        elif x > y:
            decreasing_pairs += 1
    
    total_pairs = len(lst) - 1
    if total_pairs == 0:
        return 0, 0  # Avoid division by zero for lists with fewer than 2 elements
    
    normalized_increasing = increasing_pairs / total_pairs
    normalized_decreasing = decreasing_pairs / total_pairs
    
    return normalized_increasing, normalized_decreasing


class OHM_PROBE:

    def __init__(self, param, ohm):
    
        self.param = param              
        self.ohm = ohm

        self.featuresByLayer = ['mean', 'variance', 'incPairs', 'decPairs']

        self.statsByLayer = dict()
        for f in self.featuresByLayer:
            self.statsByLayer[f] = dict()


    def AnalyzeLayer(self, layerIndex):
        
        results = self.ohm.stackMem[layerIndex].GetLSBInts()

        mean = sum(results) / len(results)
        variance = sum((x - mean) ** 2 for x in results) / len(results)
        self.statsByLayer['mean'][layerIndex] = mean
        self.statsByLayer['variance'][layerIndex] = variance

        incPairs, decPairs = count_monotonic_pairs(results)
        self.statsByLayer['incPairs'][layerIndex] = incPairs
        self.statsByLayer['decPairs'][layerIndex] = decPairs

    def AnalyzeRun(self):
        numLayers = len(self.ohm.stackMem)
        print(f"Analyzing {numLayers} layers")
        # process each layer
        for li in range(numLayers):            
            self.AnalyzeLayer(li)

        self.PlotByLayer()            

    def AnalyzeSample(self, sample):

        pass


    def PlotByLayer(self):
        
        plt1 = ['mean', 'variance']
        plt2 = ['incPairs', 'decPairs']

        for f in plt1:
            plt.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        plt.legend()
        plt.show()

        for f in plt2:
            plt.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        plt.legend()
        plt.show()
