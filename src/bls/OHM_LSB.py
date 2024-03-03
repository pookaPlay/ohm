from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_LSB:


    def __init__(self,  NN, memD) -> None:        
                
        self.N = NN
        self.memD = memD
        
        self.adders = [ADD() for _ in range(self.N)]        

        self.inIndexA = list(range(self.N))
        self.inIndexB = list(range(self.N))
        self.outIndex = list(range(self.N))

        self.Reset()          
    
    def Reset(self) -> None:
        # Connectivity
        for ai in range(self.N):
            self.adders[ai].Reset()
        
    def Output(self):        
        return self.denseOut

    def Calc(self, memA, memB, lsb=0) -> None:
    
        #memA.Output()
        denseA = memA.Output()
        denseB = memB.Output()
        self.aInputs = [memA.Output(self.inIndexA[ni]) for ni in range(self.N)]
        self.bInputs = [memB.Output(self.inIndexB[ni]) for ni in range(self.N)]

        for ai in range(self.N):
            self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)
            self.adders[ai].Print()
    
        self.sparseOut = [self.adders[ai].Output() for ai in range(self.N)]
        
        self.denseOut = list(self.memD * [0])                
        for ni in range(len(self.sparseOut)):
            self.denseOut[self.outIndex[ni]] = self.sparseOut[ni]
        

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        pass
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_LSB: {self.N} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)
        
