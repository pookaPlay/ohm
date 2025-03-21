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
        
        self.pbf = PTF(param)
        self.ibf = PTF(param)
        self.ibf.SetOneLeft()

        self.msb2lsb = msb2lsb()
        self.done = 0          
        
        self.Reset()
        
    def Reset(self) -> None:
        
        for i in range(self.d):

            self.addp[i].Reset()
            self.addn[i].Reset()
            self.lsb2msb[i].Reset()
            self.lsb2msb[i+self.d].Reset()
            
        self.flags = list(self.d2 * [0])                        

        self.pbf.Reset()
        self.ibf.Reset()

        self.msb2lsb.Reset()                
        self.done = 0
        self.debug = 0
        self.debugTicksTaken = 0
        
                    
    def lsbOut(self) -> int:
        return self.msb2lsb.SwitchStep()

    def Output(self) -> int:
        return self.msb2lsb.Output()

    def pbfOut(self):
        return self.pbf.Output()
    
    ## Combinatorial stuff goes here
    #  lsb should be a vec like x
    def Calc(self, x, wp, wn, lsb) -> None:        
        #print(f"OHM CALC")
        nx = [1-x[i] for i in range(len(x))]
        #print(f"  {x} and {nx}")
        anyLsb = max(lsb)

        for i in range(self.d):
            ni = i + self.d

            self.addp[i].Calc(x[i], wp[i], lsb[i]) 
            self.addn[i].Calc(nx[i], wn[i], lsb[i])

            if lsb[i] == 1:
                self.lsb2msb[i].Switch()                                            
                self.lsb2msb[ni].Switch()

                self.flags[i] = 0
                self.flags[ni] = 0
                # debug
                if i == 0:
                    self.debug = 0

        self.debug = self.debug + 1
        # Get the inputs for the PBF
        inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        #print(f" PBF inputs: {inputs}")

        # Calc PBF
        self.pbf.Calc(inputs, anyLsb)        
        
        if self.pbf.OutputStep() == 1:
            for i in range(self.d2):
                if self.flags[i] == 0:
                    if inputs[i] != self.pbf.Output():
                        self.flags[i] = 1                    

        self.done = 0
        self.ibf.Calc(self.flags, anyLsb)

        if self.param["debugDone"] == 1:
            if self.debug == self.param["K"]:
                print(f"   OHM DEBUG DONE")
                self.msb2lsb.SetSwitchNext()
                self.done = 1
        else:            
            if (self.ibf.Output()):            
                print(f"   OHM DONE!!!!!!!!")                
                self.debugTicksTaken = self.debug  # has ticks so save me
                self.msb2lsb.SetSwitchNext()
                self.done = 1
                self.flags = list(self.d2 * [0])
        
    # State stuff goes here
    def Step(self) -> None:        

        self.msb2lsb.Step(self.pbf.Output(), self.pbf.OutputStep())               

        self.pbf.Step()
        self.ibf.Step()

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
        print(f" = Output =====")
        self.pbf.Print()
        print(f" FLG: {self.flags} -> {self.done}")
        #print(f"  PBF: {str(inputs)} -> {self.pbf.Output()}")        
        self.msb2lsb.Print(prefix)        

