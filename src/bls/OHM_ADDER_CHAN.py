from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from OHM_ADDER_TREE import OHM_ADDER_TREE

class OHM_ADDER_CHAN(OHM_ADDER_TREE):

    def __init__(self,  numInputs, memD) -> None:        
        super().__init__(numInputs, numInputs, memD)
            

    def Reset(self) -> None:        
        super().Reset()
                
    def Calc(self, memA, memB, lsb=0) -> None:    
        super().Calc(memA, memB, lsb)        
        

    def Step(self) -> None:
        super().Step()        
