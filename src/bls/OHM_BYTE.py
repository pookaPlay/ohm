##############################
## This has the connectivity 

from BSMEM import BSMEM
from OHM_LSB import OHM_LSB

class OHM_BYTE:

    def __init__(self, memD, memK, numNodes, nodeD, input = [7, -2, -6], weights = [0]):
    
        self.NN = numNodes      # number of parallel nodes
        self.memD = memD
        self.nodeD = nodeD
        self.K = memK
        self.lsbMem = [BSMEM(self.memD, self.K), BSMEM(self.memD, self.K)]
        self.msbMem = [BSMEM(self.memD, self.K), BSMEM(self.memD, self.K)]        
        
        self.writei = 0
        self.readi = 1        

        
        self.dataMem = BSMEM(self.K, self.K)
        self.dataMem.Load(input)
        lweights = self.NN * weights
        self.paramMem = BSMEM(self.K, self.K)
        self.paramMem.Load(lweights)

        self.ohmLSB = OHM_LSB(self.NN, self.memD)           

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
        
        self.denseLSBOut = list(self.memD * [0])        

        
    def RunNStep(self, nsteps) -> None:      
            
            #self.Print()
      
            for ti in range(nsteps):
                print(f">>>>>>>>>>> Step {ti} ==========================================")     
                self.RunStep(ti)
                
                #self.Print()

                #self.readi = 1 - self.readi
                #self.writei = 1 - self.writei                
                #self.PrintMem()
                
                #print(f"LSB: {self.lsbMem[self.readi].GetInts()}")
                #print(f"MSB: {self.msbMem[self.writei].GetOffInts()}")
    
    def RunStep(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} =======================================")
            firstBit = 1            

            self.ohmLSB.Calc(self.dataMem, self.paramMem, firstBit)
            self.denseLSBOut = self.ohmLSB.Output()
 
            self.lsbMem[self.writei].Step(self.denseLSBOut)
            self.ohmLSB.Step()            
            
            
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
                
                                                  
    def GetLSBRead(self):
         return self.lsbMem[self.readi].GetLSBInts()
    
    def GetMSBRead(self):
         return self.msbMem[self.readi].GetMSBInts()
    
    def GetLSBWrite(self):
        return self.lsbMem[self.writei].GetLSBInts()
    
    def GetMSBWrite(self):
        return self.msbMem[self.writei].GetMSBInts()
                          
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
        print(prefix + f"  LSBOut: {self.denseLSBOut}")
        print(prefix + f"  MSBOut: {self.denseMSBOut}")
        self.ohmLSB.Print(prefix + "  ", showInput)
        self.ohmMSB.Print(prefix + "  ", showInput)
