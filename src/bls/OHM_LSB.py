from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_LSB:


    def __init__(self,  dataMem: DataReader, 
                        paramMem : list[DataReader], 
                        outMem) -> None:        
        
        self.out = outMem
        self.data = dataMem
        self.params = paramMem
        self.adders = [ADD() for _ in range(len(paramMem))]

        self.Reset()          
    
    def Reset(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Reset()
        
    def Output(self) -> None:
        result = [self.adders[ai].Output() for ai in range(len(self.adders))]
        return result            

    def Calc(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Calc(self.data.Output(), self.params[ai].Output())

        #print(str(self.data.Output()))
        #print(self.param.Output())

        #self.add.Calc(a, b)

    def Step(self) -> None:
        #dataMem.Output()
        pass
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_LSB: {self.Output()}")
        
