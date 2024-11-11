##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from bls.OHM import OHM

class OHM_NET:
    
    def __init__(self, D=2, Nin = 4, Nout = 5, ptf="") -> None:

        self.ohm = OHM(D, Nin, Nout, ptf)
        
    def Reset(self) -> None:
        self.ohm.Reset()        
                
    def msbOut(self) -> int:
        return self.ohm.msbOut()

    def lsbOut(self) -> int:
        return self.ohm.lsbOut()        

    def Output(self) -> int:
        return self.ohm.Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x, lsb, msb) -> None:        
        self.ohm.Calc(x, lsb, msb)          
        
    # State stuff goes here
    def Step(self, isLsb, isMsb) -> None:        

        self.ohm.Step(isLsb, isMsb)                
        

    def Print(self, prefix="", showInput=1) -> None:

        self.ohm.Print(prefix, showInput)
