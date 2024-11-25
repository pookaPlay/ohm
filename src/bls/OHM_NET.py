from bls.OHM import OHM
from bls.lsbSource import lsbSource
from bls.NeighborhoodUtil import get_window_indices, get_reflected_indices

def NOT(x):
    return 1 - x

class OHM_NET:
    
    def __init__(self, param) -> None:

        self.param = param
        
        self.L = param['L']     # Number of layers
        self.W = param['W']     # number of parallel nodes
        self.D = param['D']     # This is the node fan-in before mirroring
        self.K = param['K']     # This is the nominal (start) precision
        
        self.wZero = self.K * [0]
        self.wZero[self.K-1] = 0
        self.wOne = self.wZero.copy()
        self.wOne[0] = 1
        self.wOne[self.K-1] = 0
        self.wp = [[[lsbSource(self.K, self.wZero) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]
        self.wn = [[[lsbSource(self.K, self.wOne) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]

        self.ohm = [[OHM(param) for wi in range(self.W)] for li in range(self.L)]

        
        # 1D convolution with wrapping
        self.idx = [get_window_indices(wi, self.D, self.D) for wi in range(self.W)]        
        #for i in range(self.W):                
        #    if self.D == self.W:  # shift left
        #        self.idx[i] = int(self.idx[i][int(self.W/2):] + self.idx[i][:int(self.W/2)])
        print(f"OHM_NET WINDOW INDICES\n{self.idx}")
        
        
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
        print(f"OHM_NET CALC")        
        #print(self.idx)

        for wi in range(self.W):
            print(x)
            inputs = [x[i] for i in self.idx[wi]]
            #inputs = x
            print(f"Inputs for node {wi}: {inputs}")

            for i in range(len(inputs)):
                if lsb[i] == 1:
                    self.wp[0][wi][i].Reset()
                    self.wn[0][wi][i].Reset()                

            wpin = [wpi.Output() for wpi in self.wp[0][wi]]
            wnin = [wni.Output() for wni in self.wn[0][wi]]
            self.ohm[0][wi].Calc(inputs, wpin, wnin, lsb)

        for li in range(1, self.L):
            for wi in range(self.W):
                wpin = [wpi.Output() for wpi in self.wp[li][wi]]
                wnin = [wni.Output() for wni in self.wn[li][wi]]
                self.ohm[li][wi].Calc(x, wpin, wnin, lsb)
                
        
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
        
