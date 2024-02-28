##############################
## This has the connectivity 

from BSMEM import BSMEM
from RDMEM import RDMEM
from OHM_LSB import OHM_LSB
from OHM_MSB import OHM_MSB

class OHM_WORD:

    def __init__(self, memD, memK, numNodes, nodeD):
    
        self.NN = numNodes      # number of parallel nodes
        self.memD = memD
        self.nodeD = nodeD

        self.K = memK
        self.lsbMem = [BSMEM(self.memD, self.K), BSMEM(self.memD, self.K)]
        self.msbMem = [BSMEM(self.memD, self.K), BSMEM(self.memD, self.K)]        
        
        self.writei = 0
        self.readi = 1        

        input = [7, -2, -6]        
        self.dataMem = RDMEM(input, self.K, self.K)

        weights = self.NN * [1]
        self.paramMem = RDMEM(weights, self.K, self.K)

        self.ohmLSB = OHM_LSB(self.NN, self.memD)   
        self.ohmMSB = OHM_MSB(self.NN, self.memD, self.nodeD)

        self.denseLSBOut = list(self.memD * [0])
        self.denseMSBOut = list(self.memD * [0])

        self.Reset()

    def Reset(self) -> None:

        [mem.Reset() for mem in self.lsbMem]
        [mem.Reset() for mem in self.msbMem]
        self.memState = 0 

        self.dataMem.Reset()
        self.paramMem.Reset()
        self.ohmLSB.Reset()        
        self.ohmMSB.Reset()        
        self.denseLSBOut = list(self.memD * [0])
        self.denseMSBOut = list(self.memD * [0])

        
    def RunNStep(self, nsteps) -> None:      
            
            self.PrintMem()
      
            for ti in range(nsteps):
                print(f">>>>>>>>>>> Step {ti} ==========================================")     
                self.RunStep()
                self.readi = 1 - self.readi
                self.writei = 1 - self.writei                
                self.PrintMem()
    
    def RunStep(self):            
            ti = 0                        
            #print(f"     == {ti} =======================================")
            self.ohmLSB.Calc(self.dataMem, self.paramMem)
            self.denseLSBOut = self.ohmLSB.Output()

            self.ohmMSB.Calc(self.lsbMem[self.readi])
            self.denseMSBOut = self.ohmMSB.Output()

            self.lsbMem[self.writei].Step(self.denseLSBOut)                        
            self.msbMem[self.writei].Step(self.denseMSBOut)                                    

            #self.lsbMem[self.writei].Print()
            #self.msbMem[self.writei].Print()
            
            
            for ti in range(self.K):
                #print(f"== {ti+1} ============================")
    
                self.dataMem.Step()
                self.paramMem.Step()
                self.lsbMem[self.readi].Step()
                self.msbMem[self.readi].Step()

                self.ohmLSB.Calc(self.dataMem, self.paramMem)
                self.denseLSBOut = self.ohmLSB.Output()                      
                
                self.ohmMSB.Calc(self.lsbMem[self.readi])
                self.denseMSBOut = self.ohmMSB.Output()

                self.lsbMem[self.writei].Step(self.denseLSBOut)            
                self.msbMem[self.writei].Step(self.denseMSBOut)                                        
                                                  

    def PrintMem(self):
        print(f"LSB WRITE {self.writei}")
        self.lsbMem[self.writei].Print()
        print(f"MSB WRITE {self.writei}")
        self.msbMem[self.writei].Print()
        print(f"LSB READ {self.readi}")
        self.lsbMem[self.readi].Print()
        print(f"MSB READ {self.readi}")
        self.msbMem[self.readi].Print()

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"OHM_WORD:")
        self.ohmLSB.Print(prefix + "  ", showInput)
        self.ohmMSB.Print(prefix + "  ", showInput)
