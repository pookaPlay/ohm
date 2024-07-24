from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHAN import OHM_ADDER_CHAN
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
        self.numNodes = param['numNodes']
        self.numLayers = param['numLayers']
        
        self.memD = param['memD']        
        self.K = param['memK']        
                
        #self.dataMem = BSMEM(self.memD, self.K)                                
        self.stackMem = [BSMEM(self.memD, self.K) for _ in range(self.numLayers)]
        self.dataMem = self.stackMem[0]
        self.biasMem = [[BSMEM(self.memD, self.K) for _ in range(self.numNodes)] for _ in range(self.numLayers)]

        self.paramBiasMem = [[BSMEM(self.memD, self.K) for _ in range(self.numNodes)] for _ in range(self.numLayers)]
        self.paramStackMem = [[BSMEM(self.memD, self.K) for _ in range(param['numStack'])] for _ in range(self.numLayers)]
        self.paramThreshMem = [[BSMEM(1, self.K) for _ in range(param['numStack'])] for _ in range(self.numLayers)]

        self.bias = [[OHM_ADDER_CHAN(self.numNodes, self.memD) for _ in range(self.numNodes)] for _ in range(self.numLayers)]     
        self.stack = [[STACK_BLS(self.memD, self.memD, self.K, param) for _ in range(param['numStack'])] for _ in range(self.numLayers)] 
        
        self.doneOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        self.doneIndexOut = [list(self.numStack * [-1]) for _ in range(self.numLayers)]
        
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


    def ResetIndex(self) -> None:
        [[self.stackMem.ResetIndex() for stackMem in stackMemLayer] for stackMemLayer in self.stackMem]
        [[biasMem.ResetIndex() for biasMem in biasMemLayer] for biasMemLayer in self.biasMem]
        [[paramBiasMem.ResetIndex() for paramBiasMem in paramBiasMemLayer] for paramBiasMemLayer in self.paramBiasMem]        
        [[paramStackMem.ResetIndex() for paramStackMem in paramStackMemLayer] for paramStackMemLayer in self.paramStackMem]
        [[paramThreshMem.ResetIndex() for paramThreshMem in paramThreshMemLayer] for paramThreshMemLayer in self.paramThreshMem]
        
        
    def Run(self, input, sampleIndex, param) -> None:      
        
        print(f"Early exit")
        return
        self.ResetIndex()

        assert(len(input) > 0)

        self.input = input
        self.dataMem.LoadList(self.input)

        #print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(sampleIndex)
        #for bi in range(len(self.bias)):
        #    self.biasMem[bi].Print(f"LSB {bi}")

        if sampleIndex == param['printMem']:
            self.biasMem[0].Print("INPUT")
        
        #print(f">>>>>>>>>>> MSB PASS ")
        [biasMem.ResetIndex() for biasMem in self.biasMem]            
        
        self.RunMSB(sampleIndex)                              

        self.results = self.stackMem.GetMSBInts()
        #print(f"WC: {self.stack[0].weightCount}")
        if sampleIndex == param['printMem']:        
            self.stackMem.Print("STACK")

        
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

        return self.doneOut
                
    
    def RunMSB(self, stepi=0) -> None:            
            
            ti = 0                        
            msb = 1            
            self.denseOut = self.numStack * [0]            
            self.doneOut = self.numStack * [-1]
            self.doneIndexOut = self.numStack * [-1]
            
            #self.stepCount = 0
            #self.inputPosCount = list([0])
            #self.inputNegCount = list([0])

            if self.param['printTicks'] == 1:
                print(f"     == {stepi}:{ti} ")            

            for si in range(len(self.stack)):     
                
                self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], self.paramThreshMem[si], msb, stepi)
                                                        
                if self.doneOut[si] < 0:
                    if self.stack[si].done == 1:
                        self.doneOut[si] = ti
                        self.doneIndexOut[si] = self.stack[si].doneIndex[0]

                self.denseOut[si] = self.stack[si].Output()            
                        
            # Save output
            #print(f"     == {stepi}:{ti} {self.denseOut}")
            self.stackMem.Step(self.denseOut)

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
                    print(f"     == {stepi}:{ti} ")                

                [biasMem.Step() for biasMem in self.biasMem]
                
                for si in range(len(self.stack)):                    
                    self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], self.paramThreshMem[si], msb, stepi)

                    # Update the sticky latches                            
                    if self.doneOut[si] < 0:
                        if self.stack[si].done == 1:
                            self.doneOut[si] = ti                                                        
                            self.doneIndexOut[si] = self.stack[si].doneIndex[0]

                    self.denseOut[si] = self.stack[si].Output()            

                #print(f"     == {stepi}:{ti} {self.denseOut}")
                self.stackMem.Step(self.denseOut)

                # # Get some input stats 
                # self.stepCount = self.stepCount + 1
                # for i in range(len(self.stack[0].origInputs)):
                #     if self.stack[0].origInputs[i] > 0:
                #         self.inputPosCount[i] = self.inputPosCount[i] + 1
                #     else:
                #         self.inputNegCount[i] = self.inputNegCount[i] + 1                    

                #[stack.Step() for stack in self.stack]
            
            # fix dones for duplicates
            for si in range(len(self.stack)):                        
                if self.doneOut[si] < 0:
                    self.doneOut[si] = self.K
                    dups = [i for i, flag in enumerate(self.stack[si].flags) if flag == 0]
                    self.doneIndexOut[si] = dups[0] if len(dups) > 0 else -1
                            

    def RunLSB(self, stepi=0) -> None:            
            ti = 0                        
            #print(f"     == {stepi}:{ti} ")
            lsb = 1            

            for bi in range(len(self.bias)):
                self.bias[bi].Calc(self.dataMem, self.paramBiasMem[bi], lsb)            
                self.biasMem[bi].Step(self.bias[bi].Output())
                self.bias[bi].Step()
                                    
            lsb = 0
            for ti in range(1, self.K):
                #print(f"     == {stepi}:{ti} ")                                                
                self.dataMem.Step()         
                [paramBiasMem.Step() for paramBiasMem in self.paramBiasMem]                
        
                for bi in range(len(self.bias)):
                    self.bias[bi].Calc(self.dataMem, self.paramBiasMem[bi], lsb)
                    self.biasMem[bi].Step(self.bias[bi].Output())
                    self.bias[bi].Step()                    
                

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
 