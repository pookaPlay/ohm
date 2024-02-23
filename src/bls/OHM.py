##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from CCSRL import CCSRL
from ADD import ADD
from PTF import PTF
from SRL import SRL
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb

class OHM:
    
    def __init__(self, D=2, Nin = 4, Nout = 5, ptf="") -> None:
        ## Nin is the stored precision of weights
        ## NOut is the sign extened precision
        ## Input should be two's complement with NOut bits
        ## Weights are two's complement Nin and sign extened to NOut 

        self.d = D
        self.d2 = D*2        
        self.Nin = Nin
        self.Nout = Nout
        
        # default weights        
        self.zeros = list(self.Nin * [0])
        self.one = self.zeros.copy()
        self.one[self.Nin-1] = 1                

        self.wp = [CCSRL(self.Nin, self.Nout) for _ in range(self.d)]
        self.wn = [CCSRL(self.Nin, self.Nout) for _ in range(self.d)]        
        self.addp = [ADD() for _ in range(self.d)]
        self.addn = [ADD() for _ in range(self.d)]
        
        self.lsb2msb = [lsb2msb(self.Nout) for _ in range(self.d2)]
               
        self.lsbAtPBF = SRL(Nout*2)
        self.msbAtPBF = SRL(1)

        self.lsbAtOut = SRL(1)
        self.msbAtOut = SRL(Nout*2)

        self.flags = list(self.d2 * [0])                        
        # This is a hack - i dont think I actually need to store latchInput
        self.latchInput = list(self.d2 * [0])

        self.pbf = PTF(self.d2)                
        # Some presets for debugging
        if ptf == "min":
            self.pbf.SetMin()
        elif ptf == "max":          
            self.pbf.SetMax()                           
        else:       
            self.pbf.SetMedian()

        self.msb2lsb = msb2lsb(self.Nout)
        self.done = 0
        self.doneAtInput = 0
        
    def Reset(self) -> None:
        
        for i in range(self.d):

            self.wp[i].Reset(self.zeros)
            self.addp[i].Reset()
 
            self.wn[i].Reset(self.one)
            self.addn[i].Reset()

            self.lsb2msb[i].Reset()
            self.lsb2msb[i+self.d].Reset()
            
        self.lsbAtPBF.Reset()            
        self.msbAtPBF.Reset()
        self.lsbAtOut.Reset()            
        self.msbAtOut.Reset()
        self.flags = list(self.d2 * [0])                        
        self.latchInput = list(self.d2 * [0])
        self.pbf.Reset([self.lsb2msb[i].Output() for i in range(self.d2)])        
        self.msb2lsb.Reset()        

        self.done = 0
        self.doneAtInput = 0
                
    def msbOut(self) -> int:
        return self.msbAtOut.Output()

    def lsbOut(self) -> int:
        return self.lsbAtOut.Output()
        #return self.lsbAtPBF.Output()

    def Output(self) -> int:
        return self.msb2lsb.Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x, lsb, msb) -> None:        

        nx = [1-x[i] for i in range(len(x))]

        for i in range(self.d):
            self.addp[i].Calc(x[i], self.wp[i].Output(), lsb) 
            self.addn[i].Calc(nx[i], self.wn[i].Output(), lsb)

        # Get the inputs for the PBF
        inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        # Negate if msb to convert to offset
        if self.msbAtPBF.Output() == 1:            
            inputs = [1-x for x in inputs]
            self.flags = list(self.d2 * [0])
            self.latchInput = list(self.d2 * [0])
            self.done = 0
            # print(f"Negated inputs: {inputs}")
        else:
            for i in range(self.d2):    
                if self.flags[i] == 1:
                    inputs[i] = self.latchInput[i]

        self.pbf.Calc(inputs)
        for i in range(self.d2):
            if self.flags[i] == 0:
                if inputs[i] != self.pbf.Output():
                    self.flags[i] = 1
                    self.latchInput[i] = inputs[i]
                                        
        if (sum(self.flags) == (self.d2-1)):            
            self.done = 1            
        
    # State stuff goes here
    def Step(self, isLsb, isMsb) -> None:        
                
        # Negate MSB to convert from offset back to twos complement
        if self.msbAtPBF.Output() == 1:
            self.msb2lsb.Step(1-self.pbf.Output())
        else:
            self.msb2lsb.Step(self.pbf.Output())

        if self.done == 1:
            pass
            # reset lsb 
            
        self.lsbAtPBF.Step(isLsb)
        self.msbAtPBF.Step(isMsb)
        
        self.lsbAtOut.Step(self.lsbAtPBF.Output())
        self.msbAtOut.Step(self.msbAtPBF.Output())        

        for i in range(self.d):
            self.lsb2msb[i].Step(self.addp[i].Output(), self.flags[i])
            self.lsb2msb[i+self.d].Step(self.addn[i].Output(), self.flags[i+self.d])
            
            self.addp[i].Step()  
            self.addn[i].Step()

            self.wp[i].Step()
            self.wn[i].Step()

    def Print(self, prefix="", showInput=1) -> None:
        #print(f"==============================")
        #print(f"OHM: {self.d2} inputs")
        #print(prefix + f"################### G2G: {self.done} ###################")
        if showInput:            
            print(f" +ve ------------------------")
            for i in range(self.d):                
                prefix = f"   x{i}-"
                #self.wp[i].Print(prefix)
                #self.addp[i].Print(prefix)
                self.lsb2msb[i].Print(prefix)                        
            
            print(f" -ve ------------------------")
            for i in range(self.d):                                
                prefix = f"   x{i}-"                                                
                #self.wn[i].Print(prefix)
                #self.addn[i].Print(prefix)
                self.lsb2msb[i+self.d].Print(prefix)

        prefix = "  "
        #inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        print(f" =Output =====")
        self.lsbAtPBF.Print(" LSB-atPBF-")        
        self.msbAtPBF.Print(" MSB-atPBF-")
        self.pbf.Print(" ")
        #print(f"  PBF: {str(inputs)} -> {self.pbf.Output()}")        
        self.msb2lsb.Print(prefix)
        self.lsbAtOut.Print(" LSB-OUT-")        
        self.msbAtOut.Print(" MSB-OUT-")

