from bls.OHM import OHM
from bls.lsbSource import lsbSource
from bls.NeighborhoodUtil import get_window_indices, get_reflected_indices
from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset

def NOT(x):
    return 1 - x

class OHM_NET:
    
    def __init__(self, param) -> None:

        self.param = param
        
        self.L = param['L']     # Number of layers
        self.W = param['W']     # number of parallel nodes
        self.D = param['D']     # This is the node fan-in before mirroring
        self.K = param['K']     # This is the nominal (start) precision               

        self.wZero = SerializeLSBTwos(0, self.K)
        self.wOne = SerializeLSBTwos(1, self.K)

        self.wp = [[[lsbSource(self.K, self.wZero) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]
        self.wn = [[[lsbSource(self.K, self.wOne) for _ in range(self.D)] for _ in range(self.W)] for _ in range(self.L)]
        self.ohm = [[OHM(param) for wi in range(self.W)] for li in range(self.L)]
        
        # 1D convolution with wrapping
        self.idx = [get_window_indices(wi, self.D, self.W) for wi in range(self.W)]        
        print(f"OHM_NET WINDOW INDICES\n{self.idx}")
        #for i in range(self.W):                
        #    if self.D == self.W:  # shift left
        #        self.idx[i] = int(self.idx[i][int(self.W/2):] + self.idx[i][:int(self.W/2)])
        
        
        
    def Reset(self) -> None:        
        [[[wpi.Reset() for wpi in node] for node in layer] for layer in self.wp] 
        [[[wni.Reset() for wni in node] for node in layer] for layer in self.wn] 
        [[ohmi.Reset() for ohmi in layer] for layer in self.ohm]
        
                
    def lsbOut(self):
        ret = [[ohm.lsbOut() for ohm in layer] for layer in self.ohm]
        return ret

    def Output(self):
        ret = [[ohm.Output() for ohm in layer] for layer in self.ohm]
        return ret        

    def debugOut(self):
        ret = [[ohm.debugTicksTaken for ohm in layer] for layer in self.ohm]
        return ret

    # Combinatorial stuff goes here
    def Calc(self, x, lsb) -> None:        
        print(f"OHM_NET CALC")        
        print(f"Layer: 0")
        for wi in range(self.W):            
            inputs = [x[i] for i in self.idx[wi]]            
            #print(f"Inputs for node {wi}: {inputs}")
            for i in range(len(inputs)):
                if lsb[i] == 1:
                    self.wp[0][wi][i].Reset()
                    self.wn[0][wi][i].Reset()                

            wpin = [wpi.Output() for wpi in self.wp[0][wi]]
            wnin = [wni.Output() for wni in self.wn[0][wi]]
            self.ohm[0][wi].Calc(inputs, wpin, wnin, lsb)

        for li in range(1, self.L):
            print(f"Layer: {li}")
            for wi in range(self.W):
                
                inputs = [self.ohm[li-1][i].Output() for i in self.idx[wi]]
                lsbs = [self.ohm[li-1][i].lsbOut() for i in self.idx[wi]]

                for i in range(len(inputs)):
                    if lsb[i] == 1:
                        self.wp[li][wi][i].Reset()
                        self.wn[li][wi][i].Reset()                

                wpin = [wpi.Output() for wpi in self.wp[li][wi]]
                wnin = [wni.Output() for wni in self.wn[li][wi]]
                
                self.ohm[li][wi].Calc(inputs, wpin, wnin, lsbs)
                
        
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
        
