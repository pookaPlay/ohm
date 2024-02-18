from OHM import OHM

class OhmNet1:
    
    def __init__(self, D=2, NBits=8) -> None:
        self.D = D
        self.NBits = NBits
        self.net = OHM(D, NBits)
    
    def Reset(self, x) -> None:
        self.net.Reset(x)                    

    def Output(self) -> int:
        return self.net.Output()
                
    def Step(self, x) -> None:                
        self.net.Step(x)
                
    def Print(self, prefix="", showInput=1) -> None:        
        self.net.Print()