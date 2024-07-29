from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHANNEL import OHM_ADDER_CHANNEL
from bls.STACK_BLS import STACK_BLS
from bls.BSMEM import BSMEM

import networkx as nx

def GetNegativeIndex(din, N):
    if din < N/2:
        dout = int(din + N/2)
    else:
        dout = int(din - N/2)
    return dout

class RunNetwork:

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
        self.stack = [[STACK_BLS(self.numInputs*2, self.memD, self.K, param) for _ in range(self.numStack)] for _ in range(self.numLayers)] 
        
        self.doneOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.doneIndexOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.denseOut = [list(self.numStack * [0]) for _ in range(self.numLayers)]
        
        #self.paramStackGraph = [nx.Graph() for _ in range(param['numStack'])]
        
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
        self.denseOut = [list(self.numStack * [0]) for _ in range(self.numLayers)]


    def ResetIndex(self) -> None:
        [stackMemLayer.ResetIndex() for stackMemLayer in self.stackMem]
        [[biasMem.ResetIndex() for biasMem in biasMemLayer] for biasMemLayer in self.biasMem]
        [[paramBiasMem.ResetIndex() for paramBiasMem in paramBiasMemLayer] for paramBiasMemLayer in self.paramBiasMem]        
        [[paramStackMem.ResetIndex() for paramStackMem in paramStackMemLayer] for paramStackMemLayer in self.paramStackMem]
        [[paramThreshMem.ResetIndex() for paramThreshMem in paramThreshMemLayer] for paramThreshMemLayer in self.paramThreshMem]
        
        
    def Run(self, input, sampleIndex, param) -> None:      
                
        assert(len(input) > 0)
        
        self.input = input
        self.dataMem.LoadList(self.input)

        for layerIndex in range(self.numLayers):

            #print(f"Layer {layerIndex}")            
            #print(f"   >> LSB PASS ")        
            if layerIndex == 0:
                inputMem = self.dataMem
            else:
                inputMem = self.stackMem[layerIndex-1]
            
            #print(f"Mem {layerIndex}: {inputMem}")
            #inputMem.Print(f"INPUT@{layerIndex}")

            self.RunLSB(inputMem, layerIndex, sampleIndex)                            

            #for bi in range(len(self.bias)):
            #    self.biasMem[bi].Print(f"LSB {bi}")

            #if sampleIndex == param['printMem']:
            #    self.biasMem[0].Print("INPUT")                                
            
            #self.biasMem[layerIndex][0].Print(f"BIAS@{layerIndex}-0")
            #self.biasMem[layerIndex][1].Print(f"BIAS@{layerIndex}-1")
            #self.paramBiasMem[layerIndex][0].Print(f"PARAM@{layerIndex}-0")

            #print(f"   >> MSB PASS ")
            self.RunMSB(layerIndex, sampleIndex)                              
            
            self.stackMem[layerIndex].ReverseContent()

            #self.stackMem[layerIndex].Print(f"STACK@{layerIndex}")

            #self.results = self.stackMem[layerIndex].GetMSBInts()
            self.results = self.stackMem[layerIndex].GetLSBInts()            

        
        if param['adaptThresh'] > 0:           

            for si in range(len(self.stack)):
                weights = self.paramStackMem[si].GetLSBIntsHack()
                thresh = self.paramThreshMem[si].GetLSBIntsHack()
                                        
                #print(f"Result: {self.results}")
                #print(f"Pos: {self.stack[0].posCount} Neg: {self.stack[0].negCount} Thresh: {thresh[0]}")
                if self.stack[si].posCount > self.stack[si].negCount:
                #if self.results[0] > 0:
                    thresh[0] = thresh[0] + 1
                else:
                    thresh[0] = thresh[0] - 1

                if thresh[0] < 1:
                    thresh[0] = 1
                    if param['adaptWeights'] > 0:
                        di = self.doneIndexOut[si]
                        assert(di >= 0)
                        #dii = GetNegativeIndex(di, len(weights))                
                        weights[di] = weights[di] + 1                
                        print(f"{si}: MAX (lower) at index {di}")                                
                                    

                if thresh[0] > sum(weights):
                    thresh[0] = sum(weights)
                    if param['adaptWeights'] > 0:
                        di = self.doneIndexOut[si]
                        assert(di >= 0)
                        #dii = GetNegativeIndex(di, len(weights))
                        weights[di] = weights[di] + 1
                        print(f"{si}: MIN (upper) at index {di}")

                self.paramThreshMem[si].SetLSBIntsHack(thresh)            
                self.paramStackMem[si].SetLSBIntsHack(weights)        

        return self.results
                
    
    def RunMSB(self, li=0, stepi=0) -> None:            
            
            ti = 0                        
            msb = 1            
            self.denseOut[li] = self.numStack * [0]            
            self.doneOut[li] = self.numStack * [-1]
            self.doneIndexOut[li] = self.numStack * [-1]
            
            [biasMem.ResetIndex() for biasMem in self.biasMem[li]]
            [paramStackMem.ResetIndex() for paramStackMem in self.paramStackMem[li]]
            [paramThreshMem.ResetIndex() for paramThreshMem in self.paramThreshMem[li]]
            self.stackMem[li].ResetIndex()            
            [stack.Reset() for stack in self.stack[li]]

            #self.stepCount = 0
            #self.inputPosCount = list([0])
            #self.inputNegCount = list([0])

            if self.param['printTicks'] == 1:
                print(f"     == {stepi}:{li}:{ti} ")            

            for si in range(len(self.stack[li])):     
                
                self.stack[li][si].Calc(self.biasMem[li][si], self.paramStackMem[li][si], self.paramThreshMem[li][si], msb, stepi)
                                                        
                if self.doneOut[li][si] < 0:
                    if self.stack[li][si].done == 1:
                        self.doneOut[li][si] = ti
                        self.doneIndexOut[li][si] = self.stack[li][si].doneIndex[0]

                self.denseOut[li][si] = self.stack[li][si].Output()            
                        
            # Save output
            #print(f"     == {stepi}:{ti} {self.denseOut}")
            self.stackMem[li].Step(self.denseOut[li])

            # self.stepCount = self.stepCount + 1
            # for i in range(len(self.stack[0].origInputs)):
            #     if self.stack[0].origInputs[i] > 0:    
            #         self.inputPosCount[i] = self.inputPosCount[i] + 1
            #     else:
            #         self.inputNegCount[i] = self.inputNegCount[i] + 1

            #[stack.Step() for stack in self.stack]

            #self.lsbMem.Print("LSB")
            #self.msbMem.Print("MSB")
            msb = 0                
            for ti in range(1, self.K):
                
                if self.param['printTicks'] == 1:
                    print(f"     == {stepi}:{li}:{ti} ")                


                [biasMem.Step() for biasMem in self.biasMem[li]]
                
                for si in range(len(self.stack[li])):                    
                    self.stack[li][si].Calc(self.biasMem[li][si], self.paramStackMem[li][si], self.paramThreshMem[li][si], msb, stepi)

                    # Update the sticky latches                            
                    if self.doneOut[li][si] < 0:
                        if self.stack[li][si].done == 1:
                            self.doneOut[li][si] = ti                                                        
                            self.doneIndexOut[li][si] = self.stack[li][si].doneIndex[0]

                    self.denseOut[li][si] = self.stack[li][si].Output()            

                #print(f"     == {stepi}:{ti} {self.denseOut}")
                self.stackMem[li].Step(self.denseOut[li])

                # # Get some input stats 
                # self.stepCount = self.stepCount + 1
                # for i in range(len(self.stack[0].origInputs)):
                #     if self.stack[0].origInputs[i] > 0:
                #         self.inputPosCount[i] = self.inputPosCount[i] + 1
                #     else:
                #         self.inputNegCount[i] = self.inputNegCount[i] + 1                    

                #[stack.Step() for stack in self.stack]
            
            # fix dones for duplicates
            for si in range(len(self.stack[li])):                        
                if self.doneOut[li][si] < 0:
                    self.doneOut[li][si] = self.K
                    dups = [i for i, flag in enumerate(self.stack[li][si].flags) if flag == 0]
                    self.doneIndexOut[li][si] = dups[0] if len(dups) > 0 else -1
                            

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
 