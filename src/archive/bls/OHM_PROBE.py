import matplotlib.pyplot as plt
import numpy as np
from bls.OHM_PROBE_ANALYSIS import WeightAnalysis, count_monotonic_pairs

class OHM_PROBE:

    def __init__(self, param, ohm):
    
        self.param = param              
        self.ohm = ohm
        
        self.statKeys = ['minWeightIncrease', 'maxWeightIncrease', 'biasIncrease']

        self.featuresByLayer = ['mean', 'variance', 'incPairs', 'decPairs', 'eqPairs', 
                                'mean_ticks', 'min_ticks', 'max_ticks']

        self.statsByLayer = dict()
        for f in self.featuresByLayer:
            self.statsByLayer[f] = dict()
        
        self.networkStats = dict()
        self.networkStats['minWeightIncrease'] = list()
        self.networkStats['maxWeightIncrease'] = list()
        self.networkStats['biasIncrease'] = list()


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
        print(f"---------------------------------")
        print(f" MIN: {self.networkStats['minWeightIncrease']}")
        print(f" MAX: {self.networkStats['maxWeightIncrease']}")
        print(f"BIAS: {self.networkStats['biasIncrease']}")
        print(f"---------------------------------")

    def AnalyzeRun(self, ni, pii):
        
        numLayers = len(self.ohm.stackMem)
        self.localResults = dict()        

        #print(f"Analyzing {numLayers} layers")        
        for li in range(numLayers): 
            self.localResults[li] = self.ohm.stackMem[li].GetLSBInts()
            self.AnalyzeList(li, self.localResults[li])            
                
        for stat in self.statKeys:            
            self.networkStats[stat].append(self.ohm.stats[stat])                        
        
        ####################
        ticksTaken = np.array(self.ohm.doneOut)
        #sumFlag = np.zeros_like(ticksTaken)        
        for li in range(numLayers):             
            self.statsByLayer['mean_ticks'][li] = np.mean(ticksTaken[li])
            self.statsByLayer['min_ticks'][li] = np.min(ticksTaken[li])
            self.statsByLayer['max_ticks'][li] = np.max(ticksTaken[li])

        ######################
        ### AnalyzeWeights
        self.effectiveInputs = WeightAnalysis(self.ohm)     
        
        #print(f"Effective Inputs: {self.effectiveInputs}")           
    
    def PlotByLayer(self, fignum=0):

        #######################################        
        if fignum == 0:
            fig = plt.figure(1)
            fig.set_size_inches(4, 4)        
            fig.canvas.manager.window.move(0, 600)
        else:
            fig = plt.figure(10)    
            fig.set_size_inches(4, 4)        
            fig.canvas.manager.window.move(950, 600)        

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
            fig.set_size_inches(10, 10)
            fig.canvas.manager.window.move(0, 0)
        else:
            fig = plt.figure(12)    
            fig.set_size_inches(10, 10)        
            fig.canvas.manager.window.move(950, 0)        

                
        K = self.param['memK']
        ticksTaken = self.ohm.doneOut
        assert(len(ticksTaken) == len(self.localResults))
        
        # Convert ticksTaken to a numpy array for easier manipulation
        ticksTaken = np.array(ticksTaken)
        valNetwork = np.zeros_like(ticksTaken)
        sumFlag = np.zeros_like(ticksTaken)
        
        for li in range(len(ticksTaken)):
            for ni in range(len(ticksTaken[li])):
                valNetwork[li][ni] = self.localResults[li][ni]
                ticksTaken[li][ni] = ticksTaken[li][ni] #/ K                

        sumFlag = np.array(self.ohm.sumOut)
        sumFlag = sumFlag.T        
        ticksTaken = ticksTaken.T
        valNetwork = valNetwork.T

        ax3 = fig.add_subplot(331)
        ax4 = fig.add_subplot(332)
        ax1 = fig.add_subplot(333)
        ex2 = fig.add_subplot(334)
        ex1 = fig.add_subplot(335)
        ax2 = fig.add_subplot(336)
        
        sb1 = fig.add_subplot(337)
        sb2 = fig.add_subplot(338)
        sb3 = fig.add_subplot(339)
                        
        # ticks
        cax3 = ax3.imshow(ticksTaken, cmap='viridis', aspect='auto')        
        fig.colorbar(cax3, ax=ax3, orientation='vertical')        
        ax3.set_ylabel('Input')
        #ax3.set_xlabel('Layer')
        ax3.set_title('Ticks')

        # values
        cax4 = ax4.imshow(valNetwork, cmap='viridis', aspect='auto')
        fig.colorbar(cax4, ax=ax4, orientation='vertical')
        #ax4.set_ylabel('Input')
        #ax4.set_xlabel('Layer')
        ax4.set_title('Values')
        
        allweights, allthresh, allbiases = self.ohm.GetPrettyParameters()
        allweights = allweights.T
        allthresh = allthresh.T
        allbiases = allbiases.T

        cex2 = ex2.imshow(allbiases, cmap='viridis', aspect='auto')
        fig.colorbar(cex2, ax=ex2, orientation='vertical')
        ex2.set_ylabel('Inputs')
        #ex2.set_xlabel('Layer')
        ex2.set_title('Biases')

        #cex1 = ex1.imshow(allweights, cmap='viridis', aspect='auto')
        cex1 = ex1.imshow(sumFlag, cmap='viridis', aspect='auto')
        fig.colorbar(cex1, ax=ex1, orientation='vertical')
        #ex1.set_ylabel('Inputs')
        ex1.set_title('SumFlag')
        #ex1.set_title('Weights')

        cax1 = ax1.imshow(self.effectiveInputs, cmap='viridis', aspect='auto')        
        fig.colorbar(cax1, ax=ax1, orientation='vertical')
        #ax1.set_ylabel('#Inputs')
        #ax1.set_xlabel('Layer')
        ax1.set_title('Effective Inputs')
        
        cax2 = ax2.imshow(allthresh, cmap='viridis', aspect='auto')
        fig.colorbar(cax2, ax=ax2, orientation='vertical')
        #ax2.set_ylabel('Thresh')
        #ax2.set_xlabel('Layer')
        ax2.set_title('Thresh')
                
        plt1 = ['mean', 'variance']
        for f in plt1:
            sb1.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        sb1.legend()

        plt2 = ['incPairs', 'decPairs', 'eqPairs']
        for f in plt2:
            sb2.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        sb2.legend()

        #######################################        
        plt3 = ['mean_ticks', 'min_ticks', 'max_ticks']
        for f in plt3:
            sb3.plot(self.statsByLayer[f].keys(), self.statsByLayer[f].values(), label=f)
        sb3.legend()        


        # ax5.set_ylabel('Input')
        # ax5.set_xlabel('Layer')
        # ax5.set_title('SumFlag')
        # cax5 = ax5.imshow(sumFlag, cmap='viridis', aspect='auto')        
        # fig.colorbar(cax5, ax=ax5, orientation='vertical')

        # thresh = int(self.param['numInputs'])    
        # #print(f"Half D: {thresh}")
        # sumFlag2 = (sumFlag == thresh)

        # cax6 = ax6.imshow(sumFlag2, cmap='viridis', aspect='auto')
        # fig.colorbar(cax6, ax=ax6, orientation='vertical')
        # ax6.set_ylabel('Input')
        # ax6.set_xlabel('Layer')
        # ax6.set_title('SumFlag == halfD')
    
    def TwoConfigPlot(self, fignum):        
        self.SurfacePlots(fignum)
        #self.PlotByLayer(fignum)
