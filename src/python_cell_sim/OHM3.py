##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from ADD import ADD
from PTF import PTF
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        if ptf == "sort":
            self.pbf.SetSort(debugIndex, self.d)
        elif ptf == "min":
            self.pbf.SetMin()
        elif ptf == "max":          
            self.pbf.SetMax()                           
        else:       
            self.pbf.SetMedian()                

        self.Reset()
        
    def InitState(self, input, K, initMode=1) -> None:
        self.lsb2msbs[0].InitState(input[0], K, initMode=initMode)
        self.lsb2msbs[1].InitState(input[1], K, initMode=initMode)
        self.lsb2msbs[2].InitState(input[2], K, initMode=initMode)
        self.msb2lsb.InitState(input[1], K, initMode=initMode)
    
    def GetState(self):
        #state, lenn = self.msb2lsb.GetTransitionState()
        state, lenn = self.msb2lsb.GetReadState()
        ss = self.msb2lsb.onSwitchStep

        return state, lenn, ss

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
                if self.lsb2msbs[i].GotOutput() == 1:
                    self.latchInput[i] = self.lsb2msbs[i].Output()         
                else:       
                    self.latchInput[i] = self.lsb2msbs[i].Output()         
            else:
                if self.flags[i] == 0:
                    if self.lsb2msbs[i].GotOutput() == 1:
                        self.latchInput[i] = self.lsb2msbs[i].Output()
                    else:
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
        if self.done == 3:
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


    def Draw(self, ax=None, x=0, y=0, w=10, h=10):
        
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(x, x+w)
            ax.set_ylim(y, y+h)
            ax.set_aspect('equal')
            ax.axis('off')

        # Main Node Box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + w*0.05, y + h*0.92, f"OHM3 Node {self.debugIndex} (Done:{self.done})", fontsize=10)

        # Layout parameters
        margin = w * 0.05
        comp_h = (h * 0.8) / self.d
        comp_w = w * 0.15
        
        # Draw Input Paths (ADD + LSB2MSB)
        for i in range(self.d):
            cy = y + h - (i + 1) * comp_h - margin * 2
            
            # ADD
            self.addp[i].Draw(ax, x + margin, cy, comp_w, comp_h * 0.8)
            
            # LSB2MSB
            self.lsb2msbs[i].Draw(ax, x + margin + comp_w * 1.2, cy, comp_w, comp_h * 0.8)
            
            # Flags
            flag_color = 'red' if self.flags[i] else 'lightgray'
            flag_circle = patches.Circle((x + margin + comp_w * 2.5, cy + comp_h * 0.4), w * 0.01, color=flag_color)
            ax.add_patch(flag_circle)

        # PTF
        ptf_x = x + w * 0.55
        ptf_y = y + h * 0.2
        ptf_h = h * 0.6
        self.pbf.Draw(ax, ptf_x, ptf_y, comp_w, ptf_h)

        # MSB2LSB
        m2l_x = x + w * 0.75
        m2l_y = y + h * 0.2
        m2l_h = h * 0.6
        self.msb2lsb.Draw(ax, m2l_x, m2l_y, comp_w, m2l_h)


if  __name__ == "__main__":
    
    # test draw method
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_aspect('equal')
    ax.axis('off')

    ohm = OHM3()
    ohm.InitState([7, 5, 2], 4)    
    ohm.Draw(ax, 0, 0, 200, 200)    

    plt.show()  
