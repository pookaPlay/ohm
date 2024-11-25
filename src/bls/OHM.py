##################################################
## Bit Serial OHM Node with offset binary coding 
## QSELF: is two's complement less adventageous in bit-serial domain?

from bls.ADD import ADD
from bls.PTF import PTF
from bls.msb2lsb import msb2lsb
from bls.lsb2msb import lsb2msb


class OHM():
    
    def __init__(self, param) -> None:
        
        self.param = param
        
        self.d = param["D"]
        self.d2 = 2 * self.d

        self.addp = [ADD() for _ in range(self.d)]
        self.addn = [ADD() for _ in range(self.d)]
        
        self.lsb2msb = [lsb2msb() for _ in range(self.d2)]        

        self.flags = list(self.d2 * [0])
        self.pbf = PTF(self.d2)
        
        # Some presets for debugging
        if param["ptf"] == "min":
            self.pbf.SetMin()
        elif param["ptf"] == "max":          
            self.pbf.SetMax()                           
        else:       
            self.pbf.SetMedian()

        self.msb2lsb = msb2lsb()
        self.done = 0  
        
    def Reset(self) -> None:
        
        for i in range(self.d):

            self.addp[i].Reset()
            self.addn[i].Reset()
            self.lsb2msb[i].Reset()
            self.lsb2msb[i+self.d].Reset()
            
        self.flags = list(self.d2 * [0])                        
        self.latchInput = list(self.d2 * [0])
        self.pbf.Reset()
        self.msb2lsb.Reset()        
        self.done = 0
        
                    
    def lsbOut(self) -> int:
        return self.msb2lsb.SwitchStep()

    def Output(self) -> int:
        return self.msb2lsb.Output()

    def pbfOut(self):
        return self.pbf.Output()
    
    ## Combinatorial stuff goes here
    #  lsb should be a vec like x
    def Calc(self, x, wp, wn, lsb) -> None:        
        print(f"OHM CALC")
        nx = [1-x[i] for i in range(len(x))]

        for i in range(self.d):
            ni = i + self.d

            self.addp[i].Calc(x[i], wp[i], lsb[i]) 
            self.addn[i].Calc(nx[i], wn[i], lsb[i])

            if lsb[i] == 1:
                self.lsb2msb[i].Switch()                                            
                self.lsb2msb[ni].Switch()

                self.flags[i] = 0
                self.flags[ni] = 0

        # Get the inputs for the PBF
        inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        #print(f" PBF inputs: {inputs}")

        # Calc PBF
        self.pbf.Calc(inputs)
        self.pbf.Print(" ")
        
        for i in range(self.d2):
            if self.flags[i] == 0:
                if inputs[i] != self.pbf.Output():
                    self.flags[i] = 1                    

        # might get away with this for 4 bit
        if self.done == 0:
            if (sum(self.flags) == (self.d2-1)):            
                #print(f"===========> GOT DONE!!!!!!!!")
                self.done = 1
                self.msb2lsb.SetSwitchNext()
        else:
            self.done = 0           
        
        print(f" FLG: {self.flags} -> {self.done}")
        #self.msb2lsb.Print("M2L")        
        
    # State stuff goes here
    def Step(self) -> None:        
        
        print(f"OHM STEP")
        self.msb2lsb.Step(self.pbf.Output())               
        self.msb2lsb.Print("M2L")

        for i in range(self.d):
            self.lsb2msb[i].Step(self.addp[i].Output(), self.flags[i])             
            self.lsb2msb[i+self.d].Step(self.addn[i].Output(), self.flags[i+self.d]) 
            
            self.addp[i].Step()  
            self.addn[i].Step()

    def Print(self, prefix="", showInput=1) -> None:
        #print(f"==============================")
        #print(f"OHM: {self.d2} inputs")
        #print(prefix + f"################### G2G: {self.done} ###################")
        if showInput:            
            print(f"{prefix} +ve ------------------------")
            for i in range(self.d):                
                input = f"{prefix}   x{i}-"
                #self.wp[i].Print(prefix)
                #self.addp[i].Print(prefix)
                self.lsb2msb[i].Print(input)                        
            
            print(f"{prefix} -ve ------------------------")
            for i in range(self.d):                                
                input = f"{prefix}   x{i}-"                                                
                #self.wn[i].Print(prefix)
                #self.addn[i].Print(prefix)
                self.lsb2msb[i+self.d].Print(input)
        
        #inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        print(f"{prefix} =Output =====")
        self.pbf.Print(prefix)
        #print(f"  PBF: {str(inputs)} -> {self.pbf.Output()}")        
        self.msb2lsb.Print(prefix)        

