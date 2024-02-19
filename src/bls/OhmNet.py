from OHM import OHM

class OhmNet1:
    
    def __init__(self, D=2, NBitsIn=4, NBitsOut=5, ptf="") -> None:
    
        self.net = OHM(D, NBitsIn, NBitsOut, ptf=ptf)                
    
    def Reset(self, x) -> None:
        self.net.Reset(x)                    

    def Output(self) -> int:
        return self.net.Output()

    def Calc(self, x, lsb, msb) -> None:
        self.net.Calc(x, lsb, msb)

    def Step(self, x) -> None:                
        self.net.Step(x)
                
    def Print(self, prefix="", showInput=1) -> None:        
        self.net.Print()  