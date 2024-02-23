from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader

class OHM_LSB:


    def __init__(self,  dataMem: DataReader, 
                        paramMem : list[DataReader], 
                        outMem) -> None:        
        
        self.out = outMem
        self.data = dataMem
        self.param = paramMem
          
    
    def Reset(self) -> None:
        pass

    def Output(self) -> None:
        return

    def Calc(self) -> None:
        pass
            
    def Step(self) -> None:                
        pass
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_LSB")
        self.data.Print(prefix, verbose)
