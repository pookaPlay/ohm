from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHAN import OHM_ADDER_CHAN
from bls.STACK_ADDER_TREE import STACK_ADDER_TREE

class RunOHMS:

    def __init__(self, memD, memK, 
                 numNodes, 
                 input = [7, -2, -6], 
                 biasWeights = [0], 
                 ptfWeights = [1], 
                 ptfThresh = [0],
                 adaptWeights = 1):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes        
        #print(input)
        self.memD = memD        
        self.K = memK
        self.input = input
        self.biasWeights = biasWeights
        self.ptfWeights = ptfWeights
        self.ptfThresh = ptfThresh

        self.dataMem = BSMEM(self.memD, self.K)                

        self.stackMem = BSMEM(self.memD, self.K)                        
        self.biasMem = [BSMEM(self.memD, self.K) for _ in range(self.NN)]

        self.paramBiasMem = [BSMEM(self.memD, self.K) for _ in range(self.NN)]
        self.bias = [OHM_ADDER_CHAN(self.NN, self.memD) for _ in range(self.NN)]         
        
        self.paramStackMem = [BSMEM(self.memD, self.K) for _ in range(self.NN)]
        self.paramThreshMem = [BSMEM(1, self.K) for _ in range(self.NN)]

        self.stack = [STACK_ADDER_TREE(self.NN, self.memD, self.K, adaptWeights) for _ in range(1)] #self.NN)]
        
        self.doneOut = list(self.NN * [-1])
        self.doneIndexOut = list(self.NN * [-1])

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
            self.paramBiasMem[mi].LoadList(self.biasWeights)

        [bias.Reset() for bias in self.bias]
               
        for mi in range(len(self.paramStackMem)):
            self.paramStackMem[mi].Reset()            
            self.paramStackMem[mi].LoadList(self.ptfWeights)
            self.paramThreshMem[mi].Reset()
            self.paramThreshMem[mi].LoadList(self.ptfThresh)
        
        [stack.Reset() for stack in self.stack]


    def ResetIndex(self) -> None:
        self.stackMem.ResetIndex()
        [biasMem.ResetIndex() for biasMem in self.biasMem]                        
        [paramBiasMem.ResetIndex() for paramBiasMem in self.paramBiasMem]        
        [paramStackMem.ResetIndex() for paramStackMem in self.paramStackMem]
        [paramThreshMem.ResetIndex() for paramThreshMem in self.paramThreshMem]
        

    def ParameterUpdate(self) -> None:        
        #print("Parameter Update")
        #print(self.results)
        return
    
    def Run(self, input = [], sampleIndex = -1) -> None:      

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
        thresh = self.paramThreshMem[0].GetLSBInts()                                                        
        #print(f"       Thresh on out: {thresh}")                                       

        #if sampleIndex == printIndex:
        #    self.stackMem.Print("MSB")

        self.results = self.stackMem.GetMSBInts()

        self.ParameterUpdate()
        #if sampleIndex == printIndex:
        #    print(f"{self.results} in {self.doneOut} cycles")
        
        #print(self.doneOut)
        return self.doneOut[0]
                
    
    def RunMSB(self, stepi=0) -> None:            
            ti = 0                        
            msb = 1            
            self.denseOut = list(self.NN * [0])
            #self.doneOut = list(self.NN * [-1])
            self.doneOut = list(1 * [-1])
            self.doneIndexOut = list(1 * [-1])
            #print(f"     == {stepi}:{ti} ")            
            for si in range(len(self.stack)):     
                
                self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], self.paramThreshMem[si], msb, stepi)
                
                if self.doneOut[si] < 0:
                    if self.stack[si].done == 1:
                        self.doneOut[si] = ti
                        self.doneIndexOut[si] = self.stack[si].doneIndex

                self.denseOut[si] = self.stack[si].Output()            
                        
            # Save output
            #print(f"     == {stepi}:{ti} {self.denseOut}")
            self.stackMem.Step(self.denseOut)
            
            #[stack.Step() for stack in self.stack]

            #self.lsbMem.Print("LSB")
            #self.msbMem.Print("MSB")
            msb = 0                
            for ti in range(1, self.K):
                
                #print(f"     == {stepi}:{ti} ")                
                [biasMem.Step() for biasMem in self.biasMem]
                
                for si in range(len(self.stack)):                    
                    self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], self.paramThreshMem[si], msb, stepi)
                    
                    if self.doneOut[si] < 0:
                        if self.stack[si].done == 1:
                            self.doneOut[si] = ti
                            self.doneIndexOut[si] = self.stack[si].doneIndex[si]

                    self.denseOut[si] = self.stack[si].Output()            

                #print(f"     == {stepi}:{ti} {self.denseOut}")
                self.stackMem.Step(self.denseOut)
                                
                #[stack.Step() for stack in self.stack]
            
            # fix dones for duplicates
            for si in range(len(self.stack)):                    
                if self.doneOut[si] < 0:
                    self.doneOut[si] = self.K
                    dups = [i for i, flag in enumerate(self.stack[si].flags) if flag == 0]
                    self.doneIndexOut[si] = dups[0] if len(dups) > 0 else -1

            
           
                #self.lsbMem.Print("LSB")
                #self.msbMem.Print("MSB")


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
 