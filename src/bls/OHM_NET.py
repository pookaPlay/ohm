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
        self.wZero[self.K-1] = 1
        self.wOne = self.wZero.copy()
        self.wOne[0] = 1
        self.wOne[self.K-1] = 1
        self.wp = [[[lsbSource(self.K, self.wZero) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]
        self.wn = [[[lsbSource(self.K, self.wOne) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]

        self.ohm = [[OHM(param) for wi in range(self.W)] for li in range(self.L)]

        # 1D convolution with wrapping
        self.inputIndex = [[get_window_indices(wi, self.D, self.W) for wi in range(self.W)] for li in range(self.L)]
        if self.D == self.W:  # shift left 
            self.inputIndex = self.inputIndex[int(self.W/2):] + self.inputIndex[:int(self.W/2)]
        # 1D convolution with reflection
        ##self.inIndexA = get_reflected_indices(adderInstance, self.numInputs, self.memD)                        

        
        
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

        # self.aInputs = [memA.Output(aIndex) for aIndex in self.inIndexA]
        # # mirror inputs
        # for ai in range(self.numInputs):
        #     self.aInputs[ai+self.numInputs] = NOT(self.aInputs[ai])

        # self.bInputs = [memB.Output(bIndex) for bIndex in self.inIndexB]

        # for ai in range(len(self.adders)):
        #     self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)

        for w in range(self.W):
            wpin = [wpi.Output() for wpi in self.wp[0][w]]
            wnin = [wni.Output() for wni in self.wn[0][w]]
            self.ohm[0][w].Calc(x, wpin, wnin, lsb)

        for l in range(1, self.L):
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
        
