##############################
## This has the connectivity 

from BSMEM import BSMEM
from OHM_ADDER_TREE import OHM_ADDER_TREE

class TestByte:

    def __init__(self, memD, memK, numNodes, nodeD, 
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

        #self.ohmAdderTree = OHM_ADDER_TREE(self.NN, self.NN, self.memD)
        self.ohmAdderTree = OHM_ADDER_TREE(self.NN, 1, self.memD)

        self.Reset()

    def Reset(self) -> None:

        self.lsbMem.Reset()
        self.msbMem.Reset()
        self.dataMem.Reset()
        self.paramMem.Reset()
        self.ohmAdderTree.Reset()                

        self.dataMem.Load(self.input)        
        self.paramMem.Load(self.weights)



        
    def RunNStep(self, nsteps) -> None:      
            
            self.PrintMem()
      
            for ti in range(nsteps):
                print(f">>>>>>>>>>> Step {ti} ")     
                self.RunStep(ti)                
    
    def RunStep(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} ")
            firstBit = 1            

            self.ohmAdderTree.Calc(self.dataMem, self.paramMem, firstBit)
            self.ohmAdderTree.Step()            
            self.ohmAdderTree.Print()
            
            self.denseOut = self.ohmAdderTree.Output()
            self.lsbMem.Step(self.denseOut)
                                    
            for ti in range(self.K):
                print(f"     == {stepi}:{ti} ")
                firstBit = 0
                self.dataMem.Step()
                #self.dataMem.Print()
                self.paramMem.Step()
        
                self.ohmAdderTree.Calc(self.dataMem, self.paramMem, firstBit)
                self.denseOut = self.ohmAdderTree.Output()
                self.lsbMem.Step(self.denseOut)
                self.ohmAdderTree.Step()                                                                        
                
                                                                            
    def PrintMem(self):
        
        self.dataMem.Print("Data")
        self.paramMem.Print("Param")
        self.lsbMem.Print("Out ")

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"TestByte:")
        print(prefix + f"  addOut: {self.denseOut}")        
        self.ohmAdderTree.Print(prefix + "  ", showInput)        
