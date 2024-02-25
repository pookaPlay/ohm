from OHM import OHM
from BSMEM import BSMEM

class OHM_MSB:

    def __init__(self, N) -> None:    
        self.N = N               
        #self.adders = [ADD() for _ in range(self.N)]        
        self.Reset()                   
        
    
    def Reset(self) -> None:
        pass

    def Output(self):
        result = [0]*self.N
        return result                    

    def Calc(self, stuff) -> None:
        pass
        
            
    def Step(self) -> None:                
        pass
        
                
    def Print(self, prefix="", verbose=1) -> None:        
        pass