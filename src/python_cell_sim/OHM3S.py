##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from ADD import ADD
from PTF import PTF
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb

class OHM3S:
    
    def __init__(self, ptf="", debugDone=0, debugIndex=-1) -> None:
        self.d = 3        
        self.d2 = 2*self.d

        self.debugDone = debugDone        
        self.debugIndex = debugIndex     

        self.addp = [ADD(), ADD(), ADD()]
        self.addn = [ADD(), ADD(), ADD()]
        self.lsbPmsbs = [lsb2msb(), lsb2msb(), lsb2msb()]        
        self.lsbNmsbs = [lsb2msb(), lsb2msb(), lsb2msb()]        
        self.msb2lsb = msb2lsb()
        self.flags = list(self.d2 * [0])
        self.pbf = PTF(self.d2)
        
        # Some presets for debugging
        if ptf == "sort":
            self.pbf.SetSort(debugIndex, self.d2)
        elif ptf == "min":
            self.pbf.SetMin()
        elif ptf == "max":          
            self.pbf.SetMax()                           
        else:       
            self.pbf.SetMedian()                

        self.Reset()
        
    def InitState(self, input, K, initMode=1) -> None:
        for ni in range(self.d):
            self.lsbPmsbs[ni].InitState(input[ni], K, initMode=initMode)
            self.lsbNmsbs[ni].InitState(-input[ni], K, initMode=initMode)
        self.msb2lsb.InitState(input[1], K, initMode=initMode)
    
    def GetState(self):
        #state, lenn = self.msb2lsb.GetTransitionState()
        state, lenn = self.msb2lsb.GetReadState()
        ss = self.msb2lsb.onSwitchStep

        return state, lenn, ss


    def Reset(self) -> None:
        self.flags = list(self.d2 * [0])                        
        self.latchInput = list(self.d2 * [0])
        self.pbf.Reset()
        self.msb2lsb.Reset()        
        self.done = 0
        self.debugTicks = 0        

        for i in range(self.d):
            self.addp[i].Reset()            
            self.lsbPmsbs[i].Reset()            
            self.lsbNmsbs[i].Reset()

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
    def Calc(self, x, wp, wn, lsb) -> None:        
        #print(f"OHM CALC")

        for i in range(self.d):
            self.addp[i].Calc(x[i], wp[i], lsb[i])             
            self.addn[i].Calc(x[i], wn[i], lsb[i])             

            if lsb[i] == 1:
                self.lsbPmsbs[i].Switch()                                            
                self.flags[i] = 0                
                if self.lsbPmsbs[i].GotOutput() == 1:
                    self.latchInput[i] = self.lsbPmsbs[i].Output()                
            else:
                if self.flags[i] == 0:
                    if self.lsb2msbs[i].GotOutput() == 1:
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
            if self.debugTicks > 0:                     
                if ((self.debugTicks+1) % self.debugDone) == 0:
                    self.done = 1
                    self.msb2lsb.SetSwitchNext()
        
        # switch is we have anything to output?
        if self.done == 0:
            if self.msb2lsb.GotOutput() == 0:
                self.done = 1
                self.msb2lsb.SetSwitchNext()
        
        if self.done == 2:
            print(f"-- {self.debugIndex} DONE on tick {self.debugTicks}")
            print(f"-- FLG: {self.flags} -> {self.done}")
        
        
    # State stuff goes here
    def Step(self) -> None:        
        self.debugTicks += 1
        #if self.done:

        # Reset done
        self.done = 0

        #print(f"OHM STEP t={self.debugTicks}")        
        if self.msb2lsb.SwitchStep() == 1:
            # back to twos before entry! 
            self.msb2lsb.Step(1 - self.pbf.Output())                                       
        else:
            self.msb2lsb.Step(self.pbf.Output())
            #self.done = 0

        for i in range(self.d):
            self.lsb2msbs[i].Step(self.addp[i].Output())                                     
            self.addp[i].Step()  
            

    def Print(self, prefix="", showInput=1, showOutput=1) -> None:
        
        print(f"{prefix} - Node {self.debugIndex} -- Done {self.done}---------------------------")
        if showInput:            
            for i in range(self.d):                
                inputPrefix = f"   x{i}-"
                #self.wp[i].Print(prefix)
                #self.addp[i].Print(prefix)
                self.lsb2msbs[i].Print()                        
        if showOutput:            
            #print(f"- Output ---------")
            self.pbf.Print(" ")
            self.msb2lsb.Print()        
            print(f"------------------------------------")

