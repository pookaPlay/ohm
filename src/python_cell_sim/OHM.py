##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from ADD import ADD
from PTF import PTF
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb
from lsbSource import lsbSource
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class OHM:
    
    def __init__(self, D=3, ptf="", debugDone=0, debugIndex=-1) -> None:
        self.d = D        
        self.debugDone = debugDone        
        self.debugIndex = debugIndex     

        self.biases = [lsbSource() for _ in range(D)]
        self.addp = [ADD() for _ in range(D)] 
        self.lsb2msbs = [lsb2msb() for _ in range(D)]        
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
        
    def InitState(self, stateIndex, input, K, initMode=1) -> None:
        assert len(input) == self.d

        for i in range(self.d):
            self.lsb2msbs[i].InitState(input[i], K, initMode=initMode)
            
            if initMode == 0:
                self.biases[i].InitState(input[i], K)
            else:
                self.biases[i].InitState(0, K)
                

        self.msb2lsb.InitState(input[stateIndex], K, initMode=initMode)
    
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
    def Calc(self, x, lsb) -> None:        
        #print(f"OHM CALC")

        for i in range(self.d):            

            if lsb[i] == 1:
                self.lsb2msbs[i].Switch()                                            
                self.flags[i] = 0                
                self.latchInput[i] = self.lsb2msbs[i].Output()         
                self.biases[i].Reset()
            else:
                if self.flags[i] == 0:
                    self.latchInput[i] = self.lsb2msbs[i].Output()
                            
            self.addp[i].Calc(x[i], self.biases[i].Output(), lsb[i])                                         

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
            self.biases[i].Step()           

    def Print(self, prefix="", showInput=1, showOutput=1) -> None:
        
        print(f"{prefix} - Node {self.debugIndex} -- Done {self.done}---------------------------")
        if showInput:                        
            for i in range(self.d):                
                inputPrefix = f"   x{i}-"
                #self.wp[i].Print(prefix)
                #self.addp[i].Print(prefix)
                #bias.Print(prefix)
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
        
        # Component widths
        add_w = w * 0.08
        l2m_w = w * 0.25
        ptf_w = w * 0.08        
        gap = w * 0.02
        
        # Draw Input Paths (ADD + LSB2MSB)
        for i in range(self.d):
            cy = y + h - (i + 1) * comp_h - margin * 2
            
            # ADD
            #self.addp[i].Draw(ax, x + margin, cy + comp_h * 0.25, add_w, comp_h * 0.4)
            self.addp[i].Draw(ax, x, cy + comp_h * 0.25, add_w, comp_h * 0.4)
            
            # LSB2MSB
            self.lsb2msbs[i].Draw(ax, x + margin + add_w + gap, cy, l2m_w, comp_h * 1.5)  #0.95
            
            # Flags
            flag_color = 'red' if self.flags[i] else 'lightgray'
            flag_x = x + margin + add_w + gap + l2m_w + gap
            flag_y = cy + comp_h * 0.8
            flag_circle = patches.Circle((flag_x, flag_y), w * 0.01, color=flag_color)
            ax.add_patch(flag_circle)

            # Latch Input
            latch_w = w * 0.03
            latch_h = comp_h * 0.18

            latch_x = flag_x - latch_w / 2
            latch_y = cy + comp_h * 0.45
            
            rect_latch = patches.Rectangle((latch_x, latch_y), latch_w, latch_h, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect_latch)
            ax.text(latch_x + latch_w/2, latch_y + latch_h/2, str(self.latchInput[i]), ha='center', va='center', fontsize=8)

            # Draw a single solid horizontal line from the left offset directly to the latch center        
            #ax.plot([left_x, latch_center_x], [latch_center_y, latch_center_y], color='black', linewidth=1)            

        # Flags
        flag_color = 'red' if self.done else 'lightgray'
        flag_x = x + w * 0.67
        flag_y = y + h * 0.6
        flag_circle = patches.Circle((flag_x, flag_y), w * 0.01, color=flag_color)
        ax.add_patch(flag_circle)

        # PTF
        ptf_x = x + w * 0.55
        ptf_y = y + h * 0.4
        ptf_h = h * 0.4
        self.pbf.Draw(ax, ptf_x, ptf_y, ptf_w, ptf_h)

        # MSB2LSB
        m2l_x = x + w * 0.75
        m2l_y = y + h * 0.35
        m2l_h = h * 0.5
        m2l_w = w * 0.12
        self.msb2lsb.Draw(ax, m2l_x, m2l_y, m2l_w, m2l_h)


if  __name__ == "__main__":
    
    # test draw method
    WINDOW_WIDTH = 200
    fig, ax = plt.subplots()
    ax.set_xlim(0, WINDOW_WIDTH)
    ax.set_ylim(0, WINDOW_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')

    K = 4
    D = 3
    input = [7, 5, 2]
    ohm = OHM(D)
    ohm.InitState(0, input, K)    
    ohm.Draw(ax, 0, 0, WINDOW_WIDTH, WINDOW_WIDTH)    

    plt.show()  
