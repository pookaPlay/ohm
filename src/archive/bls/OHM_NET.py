##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from bls.OHM_v3 import OHM_v3

class OHM_NET:
    
    def __init__(self, param) -> None:

        self.param = param
        self.numStack = param['numStack']      # number of parallel nodes        
        self.numInputs = param['numInputs']
        self.numLayers = param['numLayers']

        self.memD = param['memD']        
        self.K = param['memK']  
        
        self.ohm = [[OHM_v3(self.numInputs, self.K, param) for _ in range(self.numStack)] for _ in range(self.numLayers)] 
        # D=2, Nin = 4, Nout = 5, ptf="") -> None:
        #self.ohm = OHM_v3(self.numInputs, self.K, Nout, ptf)
        
    def Reset(self) -> None:
        self.ohm.Reset()        
                
    def lsbOut(self) -> int:
        return self.ohm.lsbOut()        

    def Output(self) -> int:
        return self.ohm.Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x, lsb) -> None:        
        self.ohm.Calc(x, lsb)          
        
    # State stuff goes here
    def Step(self) -> None:        

        self.ohm.Step()                
        

    def Print(self, prefix="", showInput=1) -> None:

        self.ohm.Print(prefix, showInput)
