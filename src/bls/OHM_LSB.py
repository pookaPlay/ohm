from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_LSB:


    def __init__(self,  numNodes, numNodeInputs) -> None:        
                
        self.N = numNodes
        self.D = numNodeInputs
        
        self.adders = [ADD() for _ in range(self.N)]        

        self.memIndexA = list(range(self.D))
        self.memIndexB = list(range(self.D))

        self.Reset()          
    
    def Reset(self) -> None:
        # Connectivity
        for ai in range(self.N):
            self.adders[ai].Reset()
        
    def Output(self):
        result = [self.adders[ai].Output() for ai in range(self.N)]
        return result            

    def Calc(self, memA, memB) -> None:
        print(f"Calc: A")
        memA.Print()
        print(f"Calc: B")
        memB.Print()
        
        #memA.Output()
        #self.aInputs = [memA  .dataMem[self.dataIndex[ni]].Output() for ni in range(self.N)]
        #self.paramInputs = [self.paramMem[self.paramIndex[ni]].Output() for ni in range(self.N)]

        #for ai in range(self.N):
        #    self.adders[ai].Calc(A[ai], B[ai])

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        pass
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_LSB: {self.N} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)
        
