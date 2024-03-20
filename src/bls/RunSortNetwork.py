from bls.BSMEM import BSMEM
from bls.OHM_ADDER_CHAN import OHM_ADDER_CHAN
from bls.STACK_ADDER_TREE import STACK_ADDER_TREE

class RunSortNetwork:

    def __init__(self, memD, memK, 
                 numNodes, 
                 input = [7, -2, -6], weights = [0]):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes        

        self.memD = memD        
        self.K = memK
        self.input = input
        self.weights = weights

        self.biasMem = BSMEM(self.memD, self.K)
        self.stackMem = BSMEM(self.memD, self.K)                
        self.dataMem = BSMEM(self.memD, self.K)
        self.paramMem = BSMEM(self.memD, self.K)                
        
        self.biases = OHM_ADDER_CHAN(self.NN, self.memD)        
        
        self.paramStackMem = [BSMEM(self.NN, self.K) for _ in range(self.NN)]                        
        self.stack = [STACK_ADDER_TREE(self.NN, self.memD, self.K) for _ in range(self.NN)]
        
        self.doneOut = list(self.NN * [-1])

        self.Reset()

    def Reset(self) -> None:

        self.biasMem.Reset()
        self.stackMem.Reset()
        self.dataMem.Reset()
        self.paramMem.Reset()        

        self.biases.Reset()                
       
        self.dataMem.LoadList(self.input)        
        self.paramMem.LoadList(self.weights)
        
        for mi in range(len(self.paramStackMem)):
            self.paramStackMem[mi].Reset()
            self.paramStackMem[mi].LoadScalar(len(self.paramStackMem)-mi-1)            
                
        [stack.Reset() for stack in self.stack]


        
    def Run(self) -> None:      
            
        print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(0)
        self.biasMem.ResetIndex()
        self.biasMem.Print("LSB")
        print(f">>>>>>>>>>> MSB PASS ")
        self.RunMSB(0)
        #self.lsbMem.Print("LSB")
        self.stackMem.Print("MSB")
        print(self.doneOut)
                
    
    def RunMSB(self, stepi=0) -> None:            
            ti = 0                        
            msb = 1            
            self.denseOut = list(self.NN * [0])
            self.doneOut = list(self.NN * [-1])

            print(f"     == {stepi}:{ti} ")            
            for si in range(len(self.doneOut)):
                #self.paramStackMem[si].Print("WOSPARAM ")                
                self.stack[si].Calc(self.biasMem, self.paramStackMem[si], msb)
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
                self.biasMem.Step()
                
                for si in range(len(self.stack)):                    
                    self.stack[si].Calc(self.biasMem, self.paramStackMem[si], msb)
                    
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

            self.biases.Calc(self.dataMem, self.paramMem, lsb)            
            self.denseOut = self.biases.Output()
            #self.Print()
            self.biasMem.Step(self.denseOut)
            #self.lsbMem.Print()            
            self.biases.Step()                        
            
            lsb = 0
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                                                
                self.dataMem.Step()                
                self.paramMem.Step()
        
                self.biases.Calc(self.dataMem, self.paramMem, lsb)
                self.denseOut = self.biases.Output()
                #self.Print()
                self.biasMem.Step(self.denseOut)
                #self.lsbMem.Print()
                self.biases.Step()                                                                        
                

    def SaveState(self, filename) -> None:
        # Call the static method Load
        self.dataMem.Save(filename + "_dataMem")
        self.paramMem.Save(filename + "_paramMem")
        self.biasMem.Save(filename + "_lsbMem")
        #self.biases.Save(filename + "_biases")
            
    def LoadState(self, filename) -> None:
        self.dataMem = BSMEM.Load(filename + "_dataMem")
        self.paramMem = BSMEM.Load(filename + "_paramMem")
        self.biasMem = BSMEM.Load(filename + "_lsbMem")        

    def PrintMem(self):        
        self.dataMem.Print("Data")
        self.paramMem.Print("Param")
        self.biasMem.Print("Out ")

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"RunOHMS:")
        print(prefix + f"  lsbOut: {self.denseOut}")        
        self.biases.Print(prefix + "  ", showInput)        
 