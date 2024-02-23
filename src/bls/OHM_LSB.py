from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_LSB:


    def __init__(self,  N) -> None:        
                
        self.N = N
        self.adders = [ADD() for _ in range(self.N)]        
        self.Reset()          
    
    def Reset(self) -> None:
        for ai in range(self.N):
            self.adders[ai].Reset()
        
    def Output(self) -> None:
        result = [self.adders[ai].Output() for ai in range(self.N)]
        return result            

    def Calc(self, A, B) -> None:
        assert len(A) == self.N
        assert len(B) == self.N
        for ai in range(self.N):
            self.adders[ai].Calc(A[ai], B[ai])

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        pass
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_LSB: {self.Output()}")
        
