##############################
## This has the connectivity 

from BSMEM import BSMEM
from OHM_ADDER_TREE import OHM_ADDER_TREE

class OHM_BYTE:

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

        self.ohmAdderTree = OHM_ADDER_TREE(self.NN, self.memD)           


        self.Reset()

    def Reset(self) -> None:

        self.lsbMem.Reset()
        self.msbMem.Reset()
        self.dataMem.Reset()
        self.paramMem.Reset()
        self.ohmAdderTree.Reset()        
        self.denseLSBOut = list(self.memD * [0])        

        self.dataMem.Load(self.input)
        #lweights = self.NN * self.weights        
        self.paramMem.Load(self.weights)



        
    def RunNStep(self, nsteps) -> None:      
            
            self.PrintMem()
      
            for ti in range(nsteps):
                print(f">>>>>>>>>>> Step {ti} ==========================================")     
                self.RunStep(ti)                
    
    def RunStep(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} =======================================")
            firstBit = 1            

            self.ohmAdderTree.Calc(self.dataMem, self.paramMem, firstBit)
            #self.ohmAdderTree.Step()            
            #self.denseLSBOut = self.ohmLSB.Output()
 
            #self.lsbMem[self.writei].Step(self.denseLSBOut)
            
            
            return
            for ti in range(self.K):
                print(f"     == {stepi}:{ti} =======================================")
                firstBit = 0
                self.dataMem.Step()
                #self.dataMem.Print()
                self.paramMem.Step()
                
                self.lsbMem[self.readi].Step()

                self.ohmLSB.Calc(self.dataMem, self.paramMem, firstBit)
                self.denseLSBOut = self.ohmLSB.Output()                      

                self.lsbMem[self.writei].Step(self.denseLSBOut)
                self.ohmLSB.Step()                                                                        
                
                                                                            
    def PrintMem(self):
        
        self.dataMem.Print("Data")
        self.paramMem.Print("Param")
        #self.lsbMem.Print()        
        #self.msbMem.Print()

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"OHM_WORD:")
        print(prefix + f"  LSBOut: {self.denseLSBOut}")        
        self.ohmLSB.Print(prefix + "  ", showInput)        
