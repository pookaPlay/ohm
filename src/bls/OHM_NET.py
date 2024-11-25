from bls.OHM import OHM
from bls.lsbSource import lsbSource

class OHM_NET:
    
    def __init__(self, param) -> None:

        self.param = param
        #self.numStack = param['numStack']      # number of parallel nodes        
        self.L = param['L']
        self.W = param['W']
        self.D = param['D']  
        self.K = param['K']  
        
        self.wZero = self.K * [0]
        self.wZero[self.K-1] = 1
        self.wOne = self.wZero.copy()
        self.wOne[0] = 1
        self.wOne[self.K-1] = 1
        #print(f"Defaults in offset code: {self.wZero} and {self.wOne}")

        #self.wp = [lsbSource(self.K, self.wZero) for _ in range(self.D)]        
        #self.wn = [lsbSource(self.K, self.wOne) for _ in range(self.D)]
        self.wp = [[[lsbSource(self.K, self.wZero) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]
        self.wn = [[[lsbSource(self.K, self.wOne) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]
        self.ohm = [[OHM(param) for _ in range(self.W)] for _ in range(self.L)]
        
        
        
    def Reset(self) -> None:        
        [[[wpi.Reset() for wpi in node] for node in layer] for layer in self.wp] 
        [[[wni.Reset() for wni in node] for node in layer] for layer in self.wn] 
        [[ohmi.Reset() for ohmi in layer] for layer in self.ohm]
        
                
    def lsbOut(self) -> int:
        return self.ohm[0][0].lsbOut()        

    def Output(self) -> int:
        return self.ohm[0][0].Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x, lsb) -> None:        
        for l in range(self.L):
            for w in range(self.W):
                wpin = [wpi.Output() for wpi in self.wp[l][w]]
                wnin = [wni.Output() for wni in self.wn[l][w]]
                self.ohm[l][w].Calc(x, wpin, wnin, lsb)
                
        
    # State stuff goes here
    def Step(self) -> None:        
        for l in range(self.L):
            for w in range(self.W):                                
                [wpi.Step() for wpi in self.wp[l][w]]
                [wni.Step() for wni in self.wn[l][w]]                
                self.ohm[l][w].Step()
        

    def Print(self, prefix="", showInput=1) -> None:
        print(f"{prefix}OHM_NET:")
        print(f"{prefix}  L: {self.L} W: {self.W} D: {self.D} K: {self.K}")
        [[ohmi.Print(prefix + "  ", showInput) for ohmi in layer] for layer in self.ohm]
        
