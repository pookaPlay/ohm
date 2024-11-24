from bls.OHM import OHM
from bls.lsbSource import lsbSource

class OHM_NET:
    
    def __init__(self, param) -> None:

        self.param = param
        #self.numStack = param['numStack']      # number of parallel nodes        
        #self.numInputs = param['numInputs']
        #self.numLayers = param['numLayers']

        self.D = param['D']  
        self.K = param['K']  
        
        wZero = self.K * [0]
        wZero[self.K-1] = 1
        wOne = wZero.copy()
        wOne[0] = 1
        wOne[self.K-1] = 1
        print(f"Defaults in offset code: {wZero} and {wOne}")

        self.wp = [lsbSource(self.K, wZero) for _ in range(self.D)]        
        self.wn = [lsbSource(self.K, wOne) for _ in range(self.D)]        
        self.ohm = OHM(param)
        #self.ohm = [[OHM(self.numInputs, self.K, param) for _ in range(self.numStack)] for _ in range(self.numLayers)] 
        
        
    def Reset(self) -> None:
        [wpi.Reset() for wpi in self.wp]
        [wni.Reset() for wni in self.wn]

        self.ohm.Reset()        
                
    def lsbOut(self) -> int:
        return self.ohm.lsbOut()        

    def Output(self) -> int:
        return self.ohm.Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x, lsb) -> None:        

        wpin = [wpi.Output() for wpi in self.wp]
        wnin = [wni.Output() for wni in self.wn]

        self.ohm.Calc(x, wpin, wnin, lsb)
        
        
    # State stuff goes here
    def Step(self) -> None:        

        [wpi.Step() for wpi in self.wp]
        [wni.Step() for wni in self.wn]

        self.ohm.Step()                
        

    def Print(self, prefix="", showInput=1) -> None:

        self.ohm.Print(prefix, showInput)
