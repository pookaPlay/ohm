##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from ADD import ADD
from PTF import PTF
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb

class OHM3:
    
    def __init__(self, ptf="", debugDone=0, debugIndex=-1) -> None:
        self.d = 3        
        self.debugDone = debugDone        
        self.debugIndex = debugIndex     

        self.addp = [ADD(), ADD(), ADD()] 
        self.lsb2msbs = [lsb2msb(), lsb2msb(), lsb2msb()]        
        self.msb2lsb = msb2lsb()
        self.flags = list(self.d * [0])
        self.pbf = PTF(self.d)
        
        # Some presets for debugging
        if ptf == "min":
            self.pbf.SetMin()
        elif ptf == "max":          
            self.pbf.SetMax()                           
        else:       
            self.pbf.SetMedian()                

        self.Reset()
        
    def InitState(self, input, K) -> None:
        self.lsb2msbs[0].InitState(input[0], K)
        self.lsb2msbs[1].InitState(input[1], K)
        self.lsb2msbs[2].InitState(input[2], K)
        self.msb2lsb.InitState(input[1], K)
    
    def GetState(self):
        state = self.msb2lsb.GetTransitionState()
        return state


    def Reset(self) -> None:
        self.flags = list(self.d * [0])                        
        self.latchInput = list(self.d * [0])
        self.pbf.Reset()
        self.msb2lsb.Reset()        
        self.done = 0
        self.debugTicks = 0        

        for i in range(self.d):
            self.addp[i].Reset()            
            self.lsb2msbs[i].Reset()            

    def doneOut(self) -> int:
        return self.done

                    
    def lsbOut(self) -> int:
        return self.msb2lsb.SwitchStep()

    def Output(self) -> int:
        return self.msb2lsb.Output()

    def pbfOut(self):
        return self.pbf.Output()
    
    ## Combinatorial stuff goes here
    #  lsb should be a vec like x
    def Calc(self, x, wp, lsb) -> None:        
        #print(f"OHM CALC")

        for i in range(self.d):
            self.addp[i].Calc(x[i], wp[i], lsb[i])             

            if lsb[i] == 1:
                self.lsb2msbs[i].Switch()                                            
                self.flags[i] = 0                
                self.latchInput[i] = self.lsb2msbs[i].Output()                
            else:
                if self.flags[i] == 0:
                    self.latchInput[i] = self.lsb2msbs[i].Output()

        # Calc PBF
        self.pbf.Calc(self.latchInput)
        #self.pbf.Print(" ")
        
        for i in range(self.d):
            if self.flags[i] == 0:
                if self.latchInput[i] != self.pbf.Output():
                    self.flags[i] = 1                    

        if self.debugDone <= 0:
            if self.done == 0:
                if (sum(self.flags) == (self.d-1)):                
                    self.done = 1
                    self.msb2lsb.SetSwitchNext()
            else:
                self.done = 0                   
            if self.done == 1:
                print(f"-- DONE --")
                print(f" FLG: {self.flags} -> {self.done}")
        else:
            if self.debugTicks > 0:                     
                if ((self.debugTicks+1) % self.debugDone) == 0:
                    self.done = 1
                    self.msb2lsb.SetSwitchNext()
            if self.done == 1:
                print(f"-- DEBUG DONE --")                
        
        
    # State stuff goes here
    def Step(self) -> None:        
        self.debugTicks += 1
        #if self.done:

        # Reset done
        self.done = 0
        #print(f"OHM STEP t={self.debugTicks}")        
        if self.msb2lsb.SwitchStep() == 1:
            self.msb2lsb.Step(1 - self.pbf.Output())               
        else:
            self.msb2lsb.Step(self.pbf.Output())

        for i in range(self.d):
            self.lsb2msbs[i].Step(self.addp[i].Output())                                     
            self.addp[i].Step()  
            

    def Print(self, prefix="", showInput=1) -> None:
        print(f"{prefix}")
        print(f"------------------------------------")
        if showInput:            
            for i in range(self.d):                
                inputPrefix = f"   x{i}-"
                #self.wp[i].Print(prefix)
                #self.addp[i].Print(prefix)
                self.lsb2msbs[i].Print()                        
        
        print(f"- Output ---------")
        self.pbf.Print(" ")
        self.msb2lsb.Print()        
        print(f"------------------------------------")

