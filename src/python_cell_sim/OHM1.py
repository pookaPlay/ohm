##################################################
## Bit Serial OHM Node with both LSB and MSB paths

from matplotlib.lines import Line2D
from ADD import ADD
from PTF import PTF
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb
import matplotlib.pyplot as plt
import matplotlib.patches as patches

WINDOW_WIDTH = 200

class OHM1:
    
    def __init__(self, ptf="", debugDone=0, debugIndex=-1) -> None:
        self.d = 3        
        self.debugDone = debugDone        
        self.debugIndex = debugIndex     

        self.flags = list(self.d * [0])
        self.latchInput = list(self.d * [0])

        self.pbf = PTF(self.d)
        self.msb2lsb = msb2lsb()

        self.addp = ADD()
        self.lsb2msb = lsb2msb()

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
        self.lsb2msb.InitState(input, K, initMode=initMode)
        self.msb2lsb.InitState(input, K, initMode=initMode)
        self.addp.Reset()
        self.pbf.Reset()
    
    def GetState(self):
        #state, lenn = self.msb2lsb.GetTransitionState()
        state, lenn = self.msb2lsb.GetReadState()
        ss = self.msb2lsb.SwitchStep()

        return state, lenn, ss

    def Reset(self) -> None:
        self.flags = list(self.d * [0])                        
        self.latchInput = list(self.d * [0])
        self.pbf.Reset()
        self.msb2lsb.Reset()        
        self.done = 0
        self.debugTicks = 0        

        self.addp.Reset()            
        self.lsb2msb.Reset()            

    def doneOut(self) -> int:
        return self.done
                    
    #def lsbOut(self) -> int:
    #    return self.lsb2msb.SwitchStep()

    def msbOut(self) -> int:
        return self.lsb2msb.SwitchStep()

    def Output(self) -> int:
        return self.lsb2msb.Output()

    def pbfOut(self):
        return self.pbf.Output()
    
    ## Combinatorial stuff goes here
    #  lsb should be a vec like x
    def Calc(self, x, lsb, wp) -> None:        
        #print(f"OHM CALC")
        assert(len(x) == self.d)
        assert(len(lsb) == self.d)

        for i in range(self.d):
            if lsb[i] == 1:                
                self.flags[i] = 0                
                self.latchInput[i] = x[i]
            else:
                if self.flags[i] == 0:
                    self.latchInput[i] = x[i]
                            
        # Calc PBF
        self.pbf.Calc(self.latchInput)

        for i in range(self.d):
            if self.flags[i] == 0:
                if self.latchInput[i] != self.pbf.Output():
                    self.flags[i] = 1                    

        #self.pbf.Print(" ")        
        self.addp.Calc(self.msb2lsb.Output(), wp, self.msb2lsb.SwitchStep())             
        #self.lsb2msb.Switch()                                            

        if self.debugDone <= 0:
            if self.done == 0:
                if (sum(self.flags) == (self.d-1)):                
                    self.done = 1
                    self.msb2lsb.SetSwitchNext()
                    self.lsb2msb.SetSwitchNext()
        else:
            if self.debugTicks > 0:                     
                if ((self.debugTicks+1) % self.debugDone) == 0:
                    self.done = 1
                    self.msb2lsb.SetSwitchNext()
                    self.lsb2msb.SetSwitchNext()
        
        # switch is we have anything to output?
        if self.done == 0:
            if self.msb2lsb.GotOutput() == 0:
                self.done = 1
                self.msb2lsb.SetSwitchNext()
                self.lsb2msb.SetSwitchNext()
        
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
        
        self.lsb2msb.Step(self.addp.Output())                                     

        self.addp.Step()                          
            

    def Print(self, prefix="", showInput=1, showOutput=1) -> None:
        
        print(f"{prefix} - Node {self.debugIndex} -- Done {self.done}---------------------------")
        if showOutput:            
            self.pbf.Print(" ")
            self.msb2lsb.Print()        
        if showInput:            
            self.addp.Print("  - ADD: ")
            self.lsb2msb.Print()                        
            print(f"------------------------------------")


    def Draw(self, ax=None):        
        x = 0
        y = 0
        w = WINDOW_WIDTH
        h = WINDOW_WIDTH

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
        l2m_w = w * 0.15
        ptf_w = w * 0.08                
        
        # Draw Input Paths (Flags and Latch Inputs)
        for i in range(self.d):
            cy = y + h - (i + 1) * comp_h - margin * 2 + 3
            
            # Flags
            flag_color = 'red' if self.flags[i] else 'lightgray'
            flag_x = x + 7
            flag_y = cy + comp_h * 0.8
            flag_circle = patches.Circle((flag_x, flag_y), w * 0.01, color=flag_color)
            ax.add_patch(flag_circle)

            # Latch Input
            latch_w = 8
            latch_h = 12

            latch_x = x + 3 
            latch_y = cy + comp_h * 0.45
            
            rect_latch = patches.Rectangle((latch_x, latch_y), latch_w, latch_h, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect_latch)
            ax.text(latch_x + latch_w/2, latch_y + latch_h/2, str(self.latchInput[i]), ha='center', va='center', fontsize=8)                        

        # Done Flag
        flag_color = 'red' if self.done else 'lightgray'
        flag_x = x + w * 0.3
        flag_y = y + h * 0.6
        flag_circle = patches.Circle((flag_x, flag_y), w * 0.01, color=flag_color)
        ax.add_patch(flag_circle)

        # PTF
        ptf_x = flag_x - 4
        ptf_y = flag_y - 20 
        self.pbf.Draw(ax, ptf_x, ptf_y, 0, 0)

        # MSB2LSB
        m2l_x = flag_x + w * 0.1
        m2l_y = y + h * 0.3
        m2l_h = h * 0.5
        m2l_w = w * 0.12
        self.msb2lsb.Draw(ax, m2l_x, m2l_y, m2l_w, m2l_h)

        # ADD on output
        add_x = flag_x + w * 0.3
        add_y = y + h * 0.5
        self.addp.Draw(ax, add_x, add_y, add_w, comp_h * 0.4)
            
        # LSB2MSB on output
        l2m_x = flag_x + w * 0.4
        l2m_y = m2l_y
        self.lsb2msb.Draw(ax, l2m_x, l2m_y, m2l_w, m2l_h)

        # Draw some lines for the 3 inputs feeding the PBF
        x1 = 13
        y1 = 54         
        x2 = 40 
        y2 = 107

        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)

        x1 = 13
        y1 = 107         
        x2 = 40 
        y2 = 107

        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)

        x1 = 13
        y1 = 160         
        x2 = 40 
        y2 = 107

        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)

        x1 = 40
        y1 = 107         
        x2 = 55 
        y2 = 107
        # connector
        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)

        x1 = 65
        y1 = 107         
        x2 = 79 
        y2 = 107
        # connector
        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)
        
        x1 = 105
        y1 = 107         
        x2 = 119
        y2 = 107
        # connector
        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)

        x1 = 128
        y1 = 107         
        x2 = 139
        y2 = 107
        # connector
        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)

        x1 = 165
        y1 = 107         
        x2 = 175
        y2 = 107
        # connector
        line = Line2D([x1, x2], [y1, y2], color='green', linewidth=1)
        ax.add_line(line)


if  __name__ == "__main__":
    
    # test draw method
    WINDOW_WIDTH = 200
    fig, ax = plt.subplots()
    ax.set_xlim(0, WINDOW_WIDTH)
    ax.set_ylim(0, WINDOW_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')

    ohm = OHM1()
    ohm.InitState(7, 4)    
    ohm.Draw(ax) 

    plt.show()  
