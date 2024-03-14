from BSMEM import BSMEM
from OHM_ADDER_CHAN import OHM_ADDER_CHAN
from PTF_ADDER_TREE import PTF_ADDER_TREE

class RunWord:

    def __init__(self, memD, memK, 
                 numNodes, nodeD, 
                 input = [7, -2, -6], weights = [0]):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes
        self.nodeD = nodeD

        self.memD = memD        
        self.K = memK
        self.input = input
        self.weights = weights

        self.biasMem = [BSMEM(self.memD, self.K) for _ in range(self.NN)]
        self.stackMem = BSMEM(self.memD, self.K)                
        self.dataMem = BSMEM(self.K, self.K)
        self.paramBiasMem = [BSMEM(self.K, self.K) for _ in range(self.NN)]
        self.paramStackMem = [BSMEM(self.K, self.K) for _ in range(self.NN)]        
        
        self.bias = [OHM_ADDER_CHAN(self.NN, self.memD) for _ in range(self.NN)]
        self.stack = [PTF_ADDER_TREE(self.NN, self.memD, self.K) for _ in range(self.NN)]
        
        self.doneOut = list(self.NN * [-1])

        self.Reset()

    def Reset(self) -> None:
        
        self.stackMem.Reset()
        self.dataMem.Reset()
        self.dataMem.LoadList(self.input)        
        
        [biasMem.Reset() for biasMem in self.biasMem]
        [paramBiasMem.Reset() for paramBiasMem in self.paramBiasMem]
        
        for bi in range(len(self.paramBiasMem)):
            weights = list(self.NN * [bi+1])
            self.paramBiasMem[bi].LoadList(weights)
        #[paramBiasMem.LoadList(self.weights) for paramBiasMem in self.paramBiasMem]

        [paramStackMem.Reset() for paramStackMem in self.paramStackMem]
        [paramStackMem.LoadList(list(self.numNodes*[1])) for paramStackMem in self.paramStackMem]
        
        [bias.Reset() for bias in self.bias]
        [stack.Reset() for stack in self.stack]

        
    def Run(self) -> None:      
            
        print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(0)
        [biasMem.ResetIndex() for biasMem in self.biasMem]
        [biasMem.Print("BIAS") for biasMem in self.biasMem]
        
        print(f">>>>>>>>>>> MSB PASS ")
        self.RunMSB(0)        
        self.stackMem.Print("STACK")
        print(self.doneOut)
                
    
    def RunMSB(self, stepi=0) -> None:            
            ti = 0                        
            msb = 1            
            self.denseOut = list(self.NN * [0])
            self.doneOut = list(self.NN * [-1])

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
            
            [stack.Step() for stack in self.stack]

            #self.lsbMem.Print("LSB")
            #self.msbMem.Print("MSB")
            msb = 0                
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                
                
                for si in range(len(self.biasMem)):
                    self.biasMem[si].Step()                    

                    self.stack[si].Calc(self.biasMem[si], self.paramStackMem[si], msb)
                    if self.doneOut[si] < 0:
                        if self.stack[si].done == 1:
                            self.doneOut[si] = ti

                    self.denseOut[si] = self.stack[si].Output()            

                #print(f"     == {stepi}:{ti} {self.denseOut}")
                self.stackMem.Step(self.denseOut)
                [stack.Step() for stack in self.stack]
                #self.lsbMem.Print("LSB")
                #self.msbMem.Print("MSB")



    def RunLSB(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} ")
            lsb = 1            

            for bi in range(len(self.bias)):                
                self.bias[bi].Calc(self.dataMem, self.paramBiasMem[bi], lsb)            
                self.denseOut = self.bias[bi].Output()
                #self.Print()
                self.biasMem[bi].Step(self.denseOut)
                #self.lsbMem.Print()            
                self.bias[bi].Step()                        
            
            lsb = 0
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                                                
                self.dataMem.Step()

                for bi in range(len(self.bias)):
                    self.paramBiasMem[bi].Step()
        
                    self.bias[bi].Calc(self.dataMem, self.paramBiasMem[bi], lsb)
                    self.denseOut = self.bias[bi].Output()
                    #self.Print()
                    self.biasMem[bi].Step(self.denseOut)
                    #self.lsbMem.Print()
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
        print(prefix + f"RunWord:")
        print(prefix + f"  lsbOut: {self.denseOut}")        
        self.bias.Print(prefix + "  ", showInput)        
 