from bls.OHM_ADDER_CHANNEL import OHM_ADDER_CHANNEL
from bls.OHM_STACK_TWOS import OHM_STACK_TWOS
from bls.BSMEM import BSMEM
import numpy as np

def GetNegativeIndex(din, N):
    if din < N/2:
        dout = int(din + N/2)
    else:
        dout = int(din - N/2)
    return dout

class OHM_NETWORK:

    def __init__(self, input, param):
    
        self.param = param
        self.input = input        

        self.numStack = param['numStack']      # number of parallel nodes        
        self.numInputs = param['numInputs']
        self.numLayers = param['numLayers']
        
        self.memD = param['memD']        
        self.K = param['memK']        
                
        self.dataMem = BSMEM(self.memD, self.K)                                
        self.stackMem = [BSMEM(self.numStack, self.K) for _ in range(self.numLayers)]        
        self.biasMem = [[BSMEM(self.numInputs*2, self.K) for _ in range(self.numStack)] for _ in range(self.numLayers)]

        self.paramBiasMem = [[BSMEM(self.numInputs*2, self.K) for _ in range(self.numStack)] for _ in range(self.numLayers)]
        self.paramStackMem = [[BSMEM(self.numInputs*2, self.K) for _ in range(self.numStack)] for _ in range(self.numLayers)]
        self.paramThreshMem = [[BSMEM(1, self.K) for _ in range(self.numStack)] for _ in range(self.numLayers)]

        self.bias = [[OHM_ADDER_CHANNEL(self.numInputs*2, self.memD, si) for si in range(self.numStack)] for _ in range(self.numLayers)]     
        self.stack = [[OHM_STACK_TWOS(self.numInputs*2, self.memD, self.K, param) for _ in range(self.numStack)] for _ in range(self.numLayers)] 
        
        self.doneOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.doneIndexOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.denseOut = [list(self.numStack * [0]) for _ in range(self.numLayers)]
        self.sumOut = [list(self.numStack * [0]) for _ in range(self.numLayers)]
        
        #self.paramStackGraph = [nx.Graph() for _ in range(param['numStack'])]
        self.stats = dict()
        self.Reset()

    def SetAdaptWeights(self, adaptWeights) -> None:
        for stackLayer in self.stack:
            for stack in stackLayer:
                stack.adaptWeights = adaptWeights
    
    def Reset(self) -> None:
        
        for stackMemLayer in self.stackMem:
            stackMemLayer.Reset()

        self.dataMem.LoadList(self.input)
        
        [[biasMem.Reset() for biasMem in biasMemLayer] for biasMemLayer in self.biasMem]                        

        for li in range(len(self.paramBiasMem)):
            for mi in range(len(self.paramBiasMem[li])):
                self.paramBiasMem[li][mi].Reset()    
                self.paramBiasMem[li][mi].LoadList(self.param['biasWeights'][mi])

        for li in range(len(self.paramStackMem)):
            for mi in range(len(self.paramStackMem[li])):
                self.paramStackMem[li][mi].Reset() 
                self.paramStackMem[li][mi].LoadList(self.param['ptfWeights'][mi])
                self.paramThreshMem[li][mi].Reset()
                self.paramThreshMem[li][mi].LoadList(self.param['ptfThresh'][mi])
        
        [[bias.Reset() for bias in biasLayer] for biasLayer in self.bias]
        [[stack.Reset() for stack in stackLayer] for stackLayer in self.stack]

        self.doneOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.doneIndexOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.sumOut = [list(self.numStack * [0]) for _ in range(self.numLayers)]
        self.denseOut = [list(self.numStack * [0]) for _ in range(self.numLayers)]
        
        self.stats = dict()

    def ResetIndex(self) -> None:
        [stackMemLayer.ResetIndex() for stackMemLayer in self.stackMem]
        [[biasMem.ResetIndex() for biasMem in biasMemLayer] for biasMemLayer in self.biasMem]
        [[paramBiasMem.ResetIndex() for paramBiasMem in paramBiasMemLayer] for paramBiasMemLayer in self.paramBiasMem]        
        [[paramStackMem.ResetIndex() for paramStackMem in paramStackMemLayer] for paramStackMemLayer in self.paramStackMem]
        [[paramThreshMem.ResetIndex() for paramThreshMem in paramThreshMemLayer] for paramThreshMemLayer in self.paramThreshMem]
        
        
    def Run(self, input, sampleIndex, param) -> None:      
                
        assert(len(input) > 0)
        
        self.stats['minWeightIncrease'] = 0
        self.stats['maxWeightIncrease'] = 0
        self.stats['biasIncrease'] = 0
        
        self.input = input
        self.dataMem.LoadList(self.input)
        # For each layer
        for li in range(self.numLayers):

            if li == 0:
                inputMem = self.dataMem
            else:
                inputMem = self.stackMem[li-1]
            
            self.RunLSB(inputMem, li, sampleIndex)
            self.lsbResult = self.stackMem[li].GetLSBInts()            
            self.RunMSB(li, sampleIndex)                              
            
            self.stackMem[li].ReverseContent()
            self.results = self.stackMem[li].GetLSBInts()            
            
            if param['printSampleLayer'] == 1:
                print(f"    {li}: {self.results}")
            
            if param['adaptThresh'] > 0:
                self.AdaptThresholds(li, param)
                
            if param['adaptBias'] > 0:
                self.AdaptBiases(li, param)

        return self.results                            

    #############################################################################
    #############################################################################
    ## ADAPT FUNCTIONS 

    def AdaptBiases(self, li, param) -> None:
        
        # Iterate through stacks and update the parameters        
        for si in range(len(self.stack[li])):

            if len(self.doneIndexOut[li][si]) > 1:
                #print(f"GOT ONE: {li}:{si} {self.doneIndexOut[li][si]}")

                biases = self.paramBiasMem[li][si].GetLSBInts()
                
                for di in range(len(self.doneIndexOut[li][si])):
                    bi = self.doneIndexOut[li][si][di]
                    #bin = GetNegativeIndex(bi, len(biases))
                    biases[bi] = biases[bi] + di
                    #biases[bin] = biases[bin] + di
                    self.stats['biasIncrease'] = self.stats['biasIncrease'] + 1

                if li < (self.numLayers - 1):
                    #nexti = li + 1
                    nexti = li
                    self.paramBiasMem[nexti][si].LoadList(biases)
                else:
                    #nexti = 0
                    nexti = li
                    self.paramBiasMem[nexti][si].LoadList(biases)

    def AdaptThresholds(self, li, param) -> None:               
        
        # Iterate through stacks and update the parameters
        for si in range(len(self.stack[li])):
            
            if param['adaptThresh'] > 0:

                weights = self.paramStackMem[li][si].GetLSBIntsHack()
                thresh = self.paramThreshMem[li][si].GetLSBIntsHack()                                        

                posUpdate = 0
                if param['adaptThreshType'] == 'ss':
                    if self.results[si] > 0:
                        posUpdate = 1
                elif param['adaptThreshType'] == 'pc':
                    if self.stack[li][si].posCount > self.stack[li][si].negCount:
                        posUpdate = 1

                if posUpdate == 1:
                    thresh[0] = thresh[0] + 1
                else:
                    thresh[0] = thresh[0] - 1

                if thresh[0] < 1:
                    thresh[0] = 1
                    if param['adaptWeights'] > 0:
                        for di in self.doneIndexOut[li][si]:
                            assert(di >= 0)
                            #dii = GetNegativeIndex(di, len(weights))                
                            weights[di] = weights[di] + 1                
                            #print(f"{li}:{si}: MAX (lower) at index {di}")                                
                            self.stats['maxWeightIncrease'] = self.stats['maxWeightIncrease'] + 1
                                    

                if thresh[0] > sum(weights):
                    thresh[0] = sum(weights)
                    if param['adaptWeights'] > 0:
                        for di in self.doneIndexOut[li][si]:
                            assert(di >= 0)
                            #dii = GetNegativeIndex(di, len(weights))
                            weights[di] = weights[di] + 1
                            #print(f"{li}:{si}: MIN (upper) at index {di}")
                            self.stats['minWeightIncrease'] = self.stats['minWeightIncrease'] + 1

                if li < (self.numLayers - 1):
                    #nexti = li + 1
                    nexti = li
                    
                    self.paramThreshMem[nexti][si].SetLSBIntsHack(thresh)            
                    if param['adaptWeights'] > 0:
                        self.paramStackMem[nexti][si].SetLSBIntsHack(weights)        
                else:
                    #nexti = 0
                    nexti = li
                    self.paramThreshMem[nexti][si].SetLSBIntsHack(thresh)            
                    if param['adaptWeights'] > 0:
                        self.paramStackMem[nexti][si].SetLSBIntsHack(weights)        

    #############################################################################
    #############################################################################
    ## RUN FUNCTIONS 

    def RunMSB(self, li=0, stepi=0) -> None:            
            
            ti = 0                        
            msb = 1            
            self.denseOut[li] = self.numStack * [0]            
            self.doneOut[li] = self.numStack * [-1]
            self.doneIndexOut[li] = self.numStack * [-1]
            self.sumOut[li] = self.numStack * [0]

            [biasMem.ResetIndex() for biasMem in self.biasMem[li]]
            [paramStackMem.ResetIndex() for paramStackMem in self.paramStackMem[li]]
            [paramThreshMem.ResetIndex() for paramThreshMem in self.paramThreshMem[li]]
            self.stackMem[li].ResetIndex()            
            [stack.Reset() for stack in self.stack[li]]

            for si in range(len(self.stack[li])):     
                
                #self.biasMem[li][si].Print(f"BIAS@{li}-{si}")

                self.stack[li][si].Calc(self.biasMem[li][si], self.paramStackMem[li][si], self.paramThreshMem[li][si], msb, stepi)
                                                        
                if self.doneOut[li][si] < 0:
                    if self.stack[li][si].done == 1:
                        self.doneOut[li][si] = ti
                        self.doneIndexOut[li][si] = self.stack[li][si].doneIndex
                        self.sumOut[li][si] = self.stack[li][si].sumFlags

                self.denseOut[li][si] = self.stack[li][si].Output()            

                if self.param['printTicks'] == 1:
                    print(f"     == {si}:{stepi}:{li}:{ti}:INPUT {self.stack[li][si].inputs} -> {self.denseOut[li][si]}")
                    print(f"     == {si}:{stepi}:{li}:{ti}:FLAGS {self.stack[li][si].flags} ")            
                    
                        
            # Save output
            #print(f"     == {stepi}:{ti} {self.denseOut}")
            self.stackMem[li].Step(self.denseOut[li])


            msb = 0                
            for ti in range(1, self.K):
                
                [biasMem.Step() for biasMem in self.biasMem[li]]
                
                for si in range(len(self.stack[li])):                    
                    self.stack[li][si].Calc(self.biasMem[li][si], self.paramStackMem[li][si], self.paramThreshMem[li][si], msb, stepi)

                    # Update the sticky latches                            
                    if self.doneOut[li][si] < 0:
                        if self.stack[li][si].done == 1:
                            self.doneOut[li][si] = ti                                                        
                            self.doneIndexOut[li][si] = self.stack[li][si].doneIndex                            

                    self.denseOut[li][si] = self.stack[li][si].Output()            

                if self.param['printTicks'] == 1:
                    print(f"     == {si}:{stepi}:{li}:{ti}:INPUT {self.stack[li][si].inputs} -> {self.denseOut[li][si]}")
                    print(f"     == {si}:{stepi}:{li}:{ti}:FLAGS {self.stack[li][si].flags} ")                                
                
                #if ti == (self.K - 1):
                #    for si in range(len(self.stack[li])):
                #        self.sumOut[li][si] = self.stack[li][si].sumFlags
                #        #if self.sumOut[li][si] == 0:
                #        #    print(f"SUMOUT: {li}:{si}: {self.sumOut[li][si]}")
                #        #    self.biasMem[li][si].Print(f"BIAS@{li}-{si}")
                
                self.stackMem[li].Step(self.denseOut[li])
            
            # case for duplicates 
            for si in range(len(self.stack[li])):                        
                if self.doneOut[li][si] < 0:
                    self.doneOut[li][si] = self.K                    
                    dups = [i for i, flag in enumerate(self.stack[li][si].flags) if flag == 0]
                    self.doneIndexOut[li][si] = dups
                    #print(f"DUPLICATE: {li}:{si} {dups}")
                    #print(self.stack[li][si].flags)
                            

    def RunLSB(self, inputMem, li=0, stepi=0) -> None:            

            ti = 0                        
            #print(f"     == {stepi}:{ti} ")
            lsb = 1            
            [paramBiasMem.ResetIndex() for paramBiasMem in self.paramBiasMem[li]]
            [biasMem.ResetIndex() for biasMem in self.biasMem[li]]
            [bias.Reset() for bias in self.bias[li]]

            for bi in range(len(self.bias[li])):
                self.bias[li][bi].Calc(inputMem, self.paramBiasMem[li][bi], lsb)            
                #print(f"biasmem input: {self.bias[li][bi].Output()}")
                #self.biasMem[li][bi].Print(f"BIAS@{li}-{bi}")
                self.biasMem[li][bi].Step(self.bias[li][bi].Output())
                self.bias[li][bi].Step()
                                    
            lsb = 0
            for ti in range(1, self.K):
                #print(f"     == {stepi}:{ti} ")                                                
                inputMem.Step()         
                [paramBiasMem.Step() for paramBiasMem in self.paramBiasMem[li]]                
        
                for bi in range(len(self.bias[li])):
                    self.bias[li][bi].Calc(inputMem, self.paramBiasMem[li][bi], lsb)
                    self.biasMem[li][bi].Step(self.bias[li][bi].Output())
                    self.bias[li][bi].Step()                    
                

    def SaveState(self, filename) -> None:
        # Call the static method Load
        self.dataMem.Save(filename + "_dataMem")
        self.paramBiasMem.Save(filename + "_paramMem")
        self.biasMem.Save(filename + "_lsbMem")
        #self.biases.Save(filename + "_biases")
            
    def LoadState(self, filename) -> None:
        self.dataMem = BSMEM.Load(filename + "_dataMem")
        self.paramBiasMem = BSMEM.Load(filename + "_paramMem")
        self.biasMem = BSMEM.Load(filename + "_lsbMem")        

    def PrintMem(self):        
        self.dataMem.Print("Data")
        self.paramBiasMem.Print("Param")
        self.biasMem.Print("Out ")

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"RunOHMS:")
        print(prefix + f"  lsbOut: {self.denseOut}")        
        self.bias.Print(prefix + "  ", showInput)        
 
    def PrintParameters(self) -> None:
        for li in range(len(self.paramStackMem)):
            for i in range(len(self.paramStackMem[li])):
                biases = self.paramBiasMem[li][i].GetLSBInts()
                weights = self.paramStackMem[li][i].GetLSBIntsHack()
                thresh = self.paramThreshMem[li][i].GetLSBIntsHack()                    
                print(f"{li}:{i}          Bias: {biases}")                                                        
                print(f"{li}:{i}       Weights: {weights}")                                       
                print(f"{li}:{i}        Thresh: {thresh}")

    def GetPrettyParameters(self):
        allbiases = np.zeros((self.numLayers, self.numStack*self.numInputs*2))
        allweights = np.zeros((self.numLayers, self.numStack*self.numInputs*2))
        allthresh = np.zeros((self.numLayers, self.numStack))

        for li in range(len(self.paramBiasMem)):
            for i in range(len(self.paramBiasMem[li])):
                biases = self.paramBiasMem[li][i].GetLSBIntsHack()
                allbiases[li][i*len(biases):(i+1)*len(biases)] = biases

        for li in range(len(self.paramStackMem)):
            for i in range(len(self.paramStackMem[li])):
                weights = self.paramStackMem[li][i].GetLSBIntsHack()
                allweights[li][i*len(weights):(i+1)*len(weights)] = weights
                                            
        for li in range(len(self.paramStackMem)):
            for i in range(len(self.paramStackMem[li])):
                thresh = self.paramThreshMem[li][i].GetLSBIntsHack()
                allthresh[li][i] = thresh[0]

        return allweights, allthresh, allbiases
    
    # def WeightAdjust(self) -> None:
    #     weights = self.ohm.paramBiasMem[0].GetLSBIntsHack()        
    #     for i in range(len(self.weights)):                    
    #         if (self.weights[i] > 1):
    #             self.weights[i] = int(math.floor(self.weights[i] / 2))
            
    #     self.ohm.paramStackMem[0].SetLSBIntsHack(self.weights)

