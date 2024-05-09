from bls.RunWeightedLattice import RunWeightedLattice
#from bls.RunOHMS import RunOHMS
import math
import pickle
import torch

class MLRunner:

    def __init__(self, nx, nxxyy, param):
    
        self.NN = param['numNodes']      # number of parallel nodes        
        self.numNodes = param['numNodes']
        self.memD = param['memD']
        self.K = param['memK']
        self.param = param
        self.plotResults = dict()
        #self.dataMax = 2**self.K - 1.0        
        self.input = self.ScaleData(nx, param['scaleTo'], param['clipAt'])
        self.xxyy = self.ScaleData(nxxyy, param['scaleTo'], param['clipAt'])

        self.first = self.input[0].tolist()        
        self.posStatsSample = list(len(self.first) * [0.0])
        self.threshStats = list(2 *[0.0])
        self.weightStats = list(len(self.first) * [0.0])
        self.ohm = RunWeightedLattice(self.first, param)


    def Run(self, param) -> None:
        print(f"Running on {len(self.input)} samples")
        ticks = list()
        self.posStackStats = 0
        self.negStackStats = 0
        self.sampleCount = 0
        self.posStack = 0
        self.negStack = 0
        
        numSamples = len(self.input)
        #self.ohm.SetAdaptWeights(adaptWeights)        
        self.plotResults['thresh'] = list()

        if param['printParameters'] == 1:
            for i in range(len(self.ohm.paramBiasMem)):
                    biases = self.ohm.paramBiasMem[i].GetLSBInts()                                                        
                    #print(f"{i}          Bias: {biases}")                                                                  
            
            for i in range(len(self.ohm.paramStackMem)):                    
                    weights = self.ohm.paramStackMem[i].GetLSBIntsHack()
                    thresh = self.ohm.paramThreshMem[i].GetLSBIntsHack()                    
                    print(f"{i}       Weights: {weights}")                                       
                    print(f"{i}        Thresh: {thresh}")


        for ni in range(numSamples):                                                
            if param['printSample'] == 1:
                print(f"Sample {ni} ------------------")
            
            #########################################################
            #########################################################
            # Run the OHM
            sample = self.input[ni].tolist()
            atick = self.ohm.Run(sample, ni, param)
            #########################################################
            
            # # Get some stats
            # posStats = list()
            # #negStats = list()            
            # for i in range(len(self.ohm.inputPosCount)):
            #     assert(self.ohm.inputPosCount[i] + self.ohm.inputNegCount[i] == self.ohm.stepCount)
            #     posStats.append(self.ohm.inputPosCount[i]/self.ohm.stepCount)
            #     #negStats.append(self.ohm.inputNegCount[i]/self.ohm.stepCount)
            # for i in range(len(posStats)):                    
            #     self.posStatsSample[i] = self.posStatsSample[i] + posStats[i]
            #     #negStats.append(self.ohm.stack[i].negCount/self.ohm.stack[i].stepCount)

            localThresh = [self.ohm.stack[0].posCount / self.ohm.stack[0].stepCount, 
                           self.ohm.stack[0].negCount / self.ohm.stack[0].stepCount]          
            
            self.threshStats[0] = self.threshStats[0] + localThresh[0]
            self.threshStats[1] = self.threshStats[1] + localThresh[1]
            
            #for i in range(len(self.weightStats)):  
            #    self.weightStats[i] = self.weightStats[i] + self.ohm.stack[0].weightCount[i]

            #print(f"T: {localThresh}")
            
            if atick < 0: 
                atick = self.K
            ticks.append(atick)            

            outIndex = self.ohm.doneIndexOut[0]
            result = self.ohm.results[0]
            resultSign = 1 if result > 0 else -1
            stackInputs = self.ohm.biasMem[0].GetLSBInts()

            #self.plotResults['thresh'].append(thresh[0])
            if resultSign > 0:
                self.posStack = self.posStack + 1
            else:
                self.negStack = self.negStack + 1
            self.sampleCount = self.sampleCount + 1    

            if param['printSample'] == 1:
                print(f"{stackInputs} -> {result}[{self.ohm.doneIndexOut[0]}] in {self.ohm.doneOut[0]}")
            
            if (param['adaptBias'] > 0):
                biases = self.ohm.paramBiasMem[0].GetLSBInts()                                                        
                assert(len(stackInputs) == len(biases))
                
                for i in range(len(stackInputs)):
                    if stackInputs[i] > biases[i]:
                        biases[i] = biases[i] + 1
                    else:
                        biases[i] = biases[i] - 1                    
                
                self.ohm.paramBiasMem[0].LoadList(biases)
                

        if param['printIteration'] == 1:
            avg = sum(ticks) / len(ticks)
            print(f"Avg Ticks: {avg}")

            # for i in range(len(self.posStatsSample)):                    
            #         self.posStatsSample[i] = self.posStatsSample[i] / numSamples
            
            self.threshStats[0] = self.threshStats[0] / numSamples
            self.threshStats[1] = self.threshStats[1] / numSamples
            
            #totalMax = max(self.weightStats)
            #for i in range(len(self.weightStats)):
            #    self.weightStats[i] = self.weightStats[i] / totalMax
            #    self.weightStats[i] = self.weightStats[i] * len(self.weightStats)/2
            print(f"Stack Stats: +ve: {self.posStack/(self.posStack+self.negStack)} -ve: {self.negStack/(self.posStack+self.negStack)}")
            #print(f"1 Rate: {self.posStatsSample}")
            print(f"  PTF Stats: +ve: {self.threshStats[0]} -ve: {self.threshStats[1]}")
            #print(f"W Stat: {self.weightStats}")

            if param['printParameters'] == 1:
                #for i in range(len(self.ohm.paramBiasMem)):
                    #biases = self.ohm.paramBiasMem[i].GetLSBInts()
                    #print(f"{i}          Bias: {biases}")                                                              
                                
                for i in range(len(self.ohm.paramStackMem)):                    
                    biases = self.ohm.paramBiasMem[i].GetLSBInts()
                    print(f"{i}          Bias: {biases}")                                    
                    weights = self.ohm.paramStackMem[i].GetLSBIntsHack()
                    thresh = self.ohm.paramThreshMem[i].GetLSBIntsHack()                    
                    print(f"{i}       Weights: {weights}")                                       
                    print(f"{i}        Thresh: {thresh}")

        
    def WeightAdjust(self) -> None:
        weights = self.ohm.paramBiasMem[0].GetLSBIntsHack()
        #self.weights = self.ohm.paramStackMem[0].GetLSBIntsHack()
        for i in range(len(self.weights)):                    
            if (self.weights[i] > 1):
                self.weights[i] = int(math.floor(self.weights[i] / 2))
            
        self.ohm.paramStackMem[0].SetLSBIntsHack(self.weights)

    def ApplyToMap(self, param) -> None:
        print(f"Running on {len(self.xxyy)} samples")
                
        #self.ohm.SetAdaptWeights(adaptWeights)        

        ticks = 0
        results = torch.zeros(len(self.xxyy))
        for ni in range(len(self.xxyy)):            
            sample = self.xxyy[ni].tolist()
            #print(f"Sample {ni}\n{sample}")
            atick = self.ohm.Run(sample, ni, param)
            results[ni] = self.ohm.results[0]
            
            if atick < 0: 
                atick = self.K
            ticks += atick

        avg = ticks / len(self.xxyy)
        print(f"Ticks: {avg}")
        return(results)

    def ScaleData(self, data, maxValOut, clipValOut) -> None:
        self.min_value = torch.min(data)
        self.max_value = torch.max(data)        
        
        self.minScale = -3.0
        self.maxScale = 3.0
        #print(f"Scaling from: {self.min_value}->{self.max_value} to {self.minScale}->{self.maxScale}")
        
        # scale 0 -> 1
        data = (data - self.minScale) / (self.maxScale - self.minScale)
        data[data < 0.0] = 0.0
        data[data > 1.0] = 1.0
        # scale -1 -> 1
        data = (data - 0.5) * 2.0
        
        data = data * maxValOut
        
        data[data > clipValOut] = clipValOut
        data[data < -clipValOut] = -clipValOut
                
        data = torch.round(data)
        data = data.int()        
        return data
    
    def SaveScale(self, filename) -> None:
        # Save self.minScale, self.maxScale
        with open(filename, 'wb') as f:
            pickle.dump((self.min_value, self.max_value), f)
    
    def LoadScale(self, filename) -> None:
        # Load self.minScale, self.maxScale
        with open(filename, 'rb') as f:
            self.min_value, self.max_value = pickle.load(f)
       
       
