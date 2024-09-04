import matplotlib.pyplot as plt
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

        self.featuresByLayer = ['mean', 'variance', 'incPairs', 'decPairs', 'eqPairs', 
                                'mean_ticks', 'min_ticks', 'max_ticks']

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

    def PrintSomeStats(self):        
        print(f"WEIGHT UPDATES")
        print(f" MIN: {self.networkStats['minWeightIncrease']}")
        print(f" MAX: {self.networkStats['maxWeightIncrease']}")

    def AnalyzeRun(self, ni, pii, results):
        # results should be last layer
        numLayers = len(self.ohm.stackMem)
        self.localResults = dict()        

        #print(f"Analyzing {numLayers} layers")        
        for li in range(numLayers): 
            self.localResults[li] = self.ohm.stackMem[li].GetLSBInts()
            self.AnalyzeList(li, self.localResults[li])            
        
        statKeys = ['minWeightIncrease', 'maxWeightIncrease']
        for stat in statKeys:            
            self.networkStats[stat].append(self.ohm.stats[stat])                        
        
        ####################
        ticksTaken = np.array(self.ohm.doneOut)
        #sumFlag = np.zeros_like(ticksTaken)        
        for li in range(numLayers):             
            self.statsByLayer['mean_ticks'][li] = np.mean(ticksTaken[li])
            self.statsByLayer['min_ticks'][li] = np.min(ticksTaken[li])
            self.statsByLayer['max_ticks'][li] = np.max(ticksTaken[li])

    def TwoConfigPlot(self, fignum):
        self.PlotByLayer(fignum)
        self.SurfacePlots(fignum)

    def PlotByLayer(self, fignum=0):

        #######################################        
        if fignum == 0:
            fig = plt.figure(1)
            fig.set_size_inches(4, 4)        
            fig.canvas.manager.window.move(0, 600)
        else:
            fig = plt.figure(10)    
            fig.set_size_inches(4, 4)        
            fig.canvas.manager.window.move(800, 600)        

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)        

        plt1 = ['mean', 'variance']
        for f in plt1:
            ax1.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        ax1.legend()

        plt2 = ['incPairs', 'decPairs', 'eqPairs']
        for f in plt2:
            ax2.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        ax2.legend()
        
        #######################################        
        plt3 = ['mean_ticks', 'min_ticks', 'max_ticks']
        for f in plt3:
            ax3.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        ax3.legend()        


    def SurfacePlots(self, fignum=0):
        #######################################################################        
        if fignum == 0:
            fig = plt.figure(2)
            fig.set_size_inches(8, 6)                                
            fig.canvas.manager.window.move(0, 0)
        else:
            fig = plt.figure(12)    
            fig.set_size_inches(8, 6)        
            fig.canvas.manager.window.move(800, 0)        

                
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
        allweights = allweights.T
        allthresh = allthresh.T
        cax1 = ax1.imshow(allweights, cmap='viridis', aspect='auto')
        cax2 = ax2.imshow(allthresh, cmap='viridis', aspect='auto')
        fig.colorbar(cax1, ax=ax1, orientation='vertical')
        fig.colorbar(cax2, ax=ax2, orientation='vertical')
        ax1.set_ylabel('Weights')
        ax1.set_xlabel('Layer')
        ax1.set_title('Weights')
        ax2.set_ylabel('Thresh')
        ax2.set_xlabel('Layer')
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
    