from OHM import OHM
from BSMEM import BSMEM

class OHM_MSB:

    def __init__(self, lsbMem: BSMEM, msbMem: BSMEM, paramMem: BSMEM) -> None:
    
        self.lsb = lsbMem
        self.msb = msbMem
        self.param = paramMem
        
        
    
    def Reset(self) -> None:
        pass

    def Output(self) -> int:
        return 0

    def Calc(self) -> None:
        pass
            
    def Step(self) -> None:                
        pass
                
    def Print(self, prefix="", verbose=1) -> None:        
        pass