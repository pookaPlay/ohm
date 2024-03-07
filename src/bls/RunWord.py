##############################
## This has the connectivity 

from BSMEM import BSMEM
from OHM_ADDER_CHAN import OHM_ADDER_CHAN
from PTF_ADDER_TREE import PTF_ADDER_TREE

class RunWord:

    def __init__(self, memD, memK, 
                 numNodes, nodeD, 
                 input = [7, -2, -6], weights = [0]):
    
        self.NN = numNodes      # number of parallel nodes        
        self.nodeD = nodeD

        self.memD = memD        
        self.K = memK
        self.input = input
        self.weights = weights

        self.lsbMem = BSMEM(self.memD, self.K)
        self.msbMem = BSMEM(self.memD, self.K)                
        self.dataMem = BSMEM(self.K, self.K)
        self.paramMem = BSMEM(self.K, self.K)
        self.ptfMem = BSMEM(self.K, self.K)
        
        self.biases = OHM_ADDER_CHAN(self.NN, self.memD)
        self.ptf = PTF_ADDER_TREE(self.NN, self.memD)
        #self.biases.Print()

        self.Reset()

    def Reset(self) -> None:

        self.lsbMem.Reset()
        self.msbMem.Reset()
        self.dataMem.Reset()
        self.paramMem.Reset()
        self.biases.Reset()                

        self.dataMem.LoadList(self.input)        
        self.paramMem.LoadList(self.weights)

        
    def Run(self) -> None:      
            
        print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(0)
        print(f">>>>>>>>>>> MSB PASS ")
        self.RunMSB(0)
                
    
    def RunMSB(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} ")
            firstBit = 1            

    def RunLSB(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} ")
            firstBit = 1            

            self.biases.Calc(self.dataMem, self.paramMem, firstBit)            
            self.denseOut = self.biases.Output()
            #self.Print()
            self.lsbMem.Step(self.denseOut)
            #self.lsbMem.Print()            

            self.biases.Step()                        

            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                
                firstBit = 0
                
                self.dataMem.Step()                
                self.paramMem.Step()
        
                self.biases.Calc(self.dataMem, self.paramMem, firstBit)
                self.denseOut = self.biases.Output()
                #self.Print()
                self.lsbMem.Step(self.denseOut)
                #self.lsbMem.Print()
                self.biases.Step()                                                                        
                

    def SaveState(self, filename) -> None:
        # Call the static method Load
        self.dataMem.Save(filename + "_dataMem")
        self.paramMem.Save(filename + "_paramMem")
        self.lsbMem.Save(filename + "_lsbMem")
        #self.biases.Save(filename + "_biases")
            
    def LoadState(self, filename) -> None:
        self.dataMem = BSMEM.Load(filename + "_dataMem")
        self.paramMem = BSMEM.Load(filename + "_paramMem")
        self.lsbMem = BSMEM.Load(filename + "_lsbMem")        

    def PrintMem(self):        
        self.dataMem.Print("Data")
        self.paramMem.Print("Param")
        self.lsbMem.Print("Out ")

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"RunWord:")
        print(prefix + f"  lsbOut: {self.denseOut}")        
        self.biases.Print(prefix + "  ", showInput)        
