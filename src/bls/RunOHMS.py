from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHAN import OHM_ADDER_CHAN
from bls.STACK_ADDER_TREE import STACK_ADDER_TREE
from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHAN import OHM_ADDER_CHAN

def GetNegativeIndex(din, N):
    if din < N/2:
        dout = int(din + N/2)
    else:
        dout = int(din - N/2)
    return dout

class RunOHMS:

    def __init__(self, input, param):
    
        self.param = param
        self.input = input

        self.numStack = param['numStack']      # number of parallel nodes        
        self.numNodes = param['numNodes']        
        #print(input)
        self.memD = param['memD']        
        self.K = param['memK']
        
        self.dataMem = BSMEM(self.memD, self.K)                

        self.stackMem = BSMEM(self.memD, self.K)                        
        self.biasMem = [BSMEM(self.memD, self.K) for _ in range(self.numNodes)]

        self.paramBiasMem = [BSMEM(self.memD, self.K) for _ in range(self.numNodes)]
        self.bias = [OHM_ADDER_CHAN(self.numNodes, self.memD) for _ in range(self.numNodes)]         
        
        self.paramStackMem = [BSMEM(self.memD, self.K) for _ in range(param['numStack'])]
        self.paramThreshMem = [BSMEM(1, self.K) for _ in range(param['numStack'])]

        self.stack = [STACK_ADDER_TREE(self.memD, self.memD, self.K, param) for _ in range(param['numStack'])] 
        
        self.doneOut = list(self.numStack * [-1])
        self.doneIndexOut = list(self.numStack * [-1])

        self.deltas = [param['ptfDeltas'] for _ in range(param['numStack'])]
        
        #self.inputPosCount = list([0])
        #self.inputNegCount = list([0])
        #self.stepCount = 0
        self.Reset()

    def SetAdaptWeights(self, adaptWeights) -> None:
        for stack in self.stack:
            stack.adaptWeights = adaptWeights
    
    def Reset(self) -> None:

        self.dataMem.Reset()
        #print("ON RESET")
        #print(self.input)
        self.dataMem.LoadList(self.input)

        self.stackMem.Reset()
        [biasMem.Reset() for biasMem in self.biasMem]                        
    
        for mi in range(len(self.paramBiasMem)):
            self.paramBiasMem[mi].Reset()        
            self.paramBiasMem[mi].LoadList(self.param['biasWeights'])

        [bias.Reset() for bias in self.bias]
               
        for mi in range(len(self.paramStackMem)):
            self.paramStackMem[mi].Reset() 
            #print(f"###################################################") 
            #print(self.param['ptfWeights'])
            self.paramStackMem[mi].LoadList(self.param['ptfWeights'][mi])
            self.paramThreshMem[mi].Reset()
            self.paramThreshMem[mi].LoadList(self.param['ptfThresh'][mi])
        
        [stack.Reset() for stack in self.stack]


    def ResetIndex(self) -> None:
        self.stackMem.ResetIndex()
        [biasMem.ResetIndex() for biasMem in self.biasMem]                        
        [paramBiasMem.ResetIndex() for paramBiasMem in self.paramBiasMem]        
        [paramStackMem.ResetIndex() for paramStackMem in self.paramStackMem]
        [paramThreshMem.ResetIndex() for paramThreshMem in self.paramThreshMem]
        
        
    def Run(self, input, sampleIndex, param) -> None:      

        printIndex = -1        

        self.ResetIndex()

        assert(len(input) > 0)

        self.input = input
        self.dataMem.LoadList(self.input)

        #print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(sampleIndex)
        #for bi in range(len(self.bias)):
        #    self.biasMem[bi].Print(f"LSB {bi}")
        #if sampleIndex == printIndex:
        #    self.biasMem[0].Print("LSB")
        
        #print(f">>>>>>>>>>> MSB PASS ")
        [biasMem.ResetIndex() for biasMem in self.biasMem]            
        
        self.RunMSB(sampleIndex)                              

        self.results = self.stackMem.GetMSBInts()
        #print(f"WC: {self.stack[0].weightCount}")
        
        if param['printMem'] > 0:
            self.biasMem[0].Print("INPUT")
            self.stackMem.Print("STACK")
        
        if param['adaptThresh'] > 0:
            #print(f"       Result: {self.results[0]} and ThreshCount: {self.stack[0].threshCount}")
            #print(f"       Thresh count: {self.stack[0].threshCount}")
            #print(f" Done Indicies: {self.doneIndexOut}")

            weights = self.paramStackMem[0].GetLSBIntsHack()
            thresh = self.paramThreshMem[0].GetLSBIntsHack()
                                    
            if self.stack[0].posCount > self.stack[0].negCount:
                thresh[0] = thresh[0] + 1
            else:
                thresh[0] = thresh[0] - 1

            if thresh[0] < 1:
                thresh[0] = 1              
                di = self.doneIndexOut[0]
                #print(self.doneIndexOut)
                assert(di >= 0)
                dii = GetNegativeIndex(di, len(weights))                

                weights[di] = weights[di] + 1
                #weights[dii] = weights[dii] + 1
                print(f"I got an underdog: {di} -> {dii}")                                
                                

            if thresh[0] > sum(weights):
                thresh[0] = sum(weights)
                di = self.doneIndexOut[0]
                assert(di >= 0)
                dii = GetNegativeIndex(di, len(weights))
                print(f"I got an runaway: {di} -> {dii}")
                weights[di] = weights[di] + 1
                #weights[dii] = weights[dii] + 1                

            self.paramThreshMem[0].SetLSBIntsHack(thresh)            
            self.paramStackMem[0].SetLSBIntsHack(weights)
        
        if param['adaptWeights'] == 1:            
            
            weights = self.paramStackMem[0].GetLSBIntsHack()
            assert(len(weights) == len(self.stack[0].weightCount))
            for i in range(len(weights)):                                    
                if self.stack[0].weightCount[i] > 0:
                    weights[i] = weights[i] + 1
                else:
                    weights[i] = weights[i] - 1
                #weights[i] = weights[i] + self.stack[0].weightCount[i]
            #self.paramStackMem[0].SetLSBIntsHack(weights)

        return self.doneOut[0]
                
    
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
 