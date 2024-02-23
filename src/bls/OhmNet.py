from OHM import OHM

class OhmNet:

    def __init__(self, D=2, NBitsIn=4, NBitsOut=5, netW = 1, netD=2) -> None:
    
        ptf="median"
        netW = 1
        netD = 2        
        #self.net = [OHM(D, NBitsIn, NBitsOut, ptf) for _ in range(netD)]        
        self.net = OHM(D, NBitsIn, NBitsOut, ptf)
    
    def Reset(self) -> None:
        self.net.Reset()                    

    def Output(self) -> int:
        return self.net.Output()

    def msbOut(self) -> int:
        return self.net.msbOut()

    def lsbOut(self) -> int:
        return self.net.lsbOut()        

    def done(self) -> int:
        return self.net.done

    def Calc(self, x, lsb, msb) -> None:
        self.net.Calc(x, lsb, msb)
            
    def Step(self, isLsb, isMsb) -> None:                
        self.net.Step(isLsb, isMsb)
                
    def Print(self, prefix="", showInput=1) -> None:        
        self.net.Print()  