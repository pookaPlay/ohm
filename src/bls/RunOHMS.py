from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHAN import OHM_ADDER_CHAN
from bls.STACK_ADDER_TREE import STACK_ADDER_TREE

class RunOHMS:

    def __init__(self, memD, memK, 
                 numNodes, 
                 input = [7, -2, -6], 
                 biasWeights = [0], 
                 ptfWeights = [1]):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes        

        self.memD = memD        
        self.K = memK
        self.input = input
        self.biasWeights = biasWeights
        self.ptfWeights = ptfWeights
        
        self.dataMem = BSMEM(self.memD, self.K)                

        self.stackMem = BSMEM(self.memD, self.K)                        
        self.biasMem = [BSMEM(self.memD, self.K) for _ in range(self.NN)]

        self.paramBiasMem = [BSMEM(self.memD, self.K) for _ in range(self.NN)]
        self.bias = [OHM_ADDER_CHAN(self.NN, self.memD) for _ in range(self.NN)]         
        
        self.paramStackMem = [BSMEM(self.NN, self.K) for _ in range(self.NN)]                        
        self.stack = [STACK_ADDER_TREE(self.NN, self.memD, self.K) for _ in range(1)] #self.NN)]
        
        self.doneOut = list(self.NN * [-1])

        self.Reset()

    def Reset(self) -> None:

        self.dataMem.Reset()
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
            if mi == 0:
                self.paramStackMem[mi].Print("PTF")

        [stack.Reset() for stack in self.stack]

        
    def Run(self) -> None:      
            
        print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(0)
        #for bi in range(len(self.bias)):
        #    self.biasMem[bi].Print(f"LSB {bi}")
        self.biasMem[0].Print("LSB")

        print(f">>>>>>>>>>> MSB PASS ")
        [biasMem.ResetIndex() for biasMem in self.biasMem]            
        
        self.RunMSB(0)
        
        self.stackMem.Print("MSB")
        print(self.doneOut)
                
    
    def RunMSB(self, stepi=0) -> None:            
            ti = 0                        
            msb = 1            
            self.denseOut = list(self.NN * [0])
            #self.doneOut = list(self.NN * [-1])
            self.doneOut = list(1 * [-1])

            print(f"     == {stepi}:{ti} ")            
            for si in range(len(self.doneOut)):                
                self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], msb)
                if self.doneOut[si] < 0:
                    if self.stack[si].done == 1:
                        self.doneOut[si] = ti

                self.denseOut[si] = self.stack[si].Output()            
                        
            # Save output
            #print(f"     == {stepi}:{ti} {self.denseOut}")
            self.stackMem.Step(self.denseOut)
            
            #[stack.Step() for stack in self.stack]

            #self.lsbMem.Print("LSB")
            #self.msbMem.Print("MSB")
            msb = 0                
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                
                [biasMem.Step() for biasMem in self.biasMem]
                
                for si in range(len(self.stack)):                    
                    self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], msb)
                    
                    if self.doneOut[si] < 0:
                        if self.stack[si].done == 1:
                            self.doneOut[si] = ti

                    self.denseOut[si] = self.stack[si].Output()            

                #print(f"     == {stepi}:{ti} {self.denseOut}")
                self.stackMem.Step(self.denseOut)
                
                #[stack.Step() for stack in self.stack]
                
                #self.lsbMem.Print("LSB")
                #self.msbMem.Print("MSB")


    def RunLSB(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} ")
            lsb = 1            

            for bi in range(len(self.bias)):
                self.bias[bi].Calc(self.dataMem, self.paramBiasMem[bi], lsb)            
                self.biasMem[bi].Step(self.bias[bi].Output())
                self.bias[bi].Step()
                                    
            lsb = 0
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                                                
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
 