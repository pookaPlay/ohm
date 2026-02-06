from DataIO import SerializeMSBTwos, SerializeLSBTwos, SerializeMSBOffset, DeserializeLSBTwos, DeserializeMSBTwos, DeserializeLSBOffset, DeserializeMSBOffset
import matplotlib.patches as patches
from matplotlib.lines import Line2D

class msb2lsb:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0                
        self.onSwitchStep = 1
        self.switchNext = 0
        self.lastVal = 0
        #self.transitionState = self.GetReadState()

    def InitState(self, input, K, initMode=1) -> None:
        readMode = 1 - self.mode
        myZero = [0] * K

        if initMode == 0:            
            self.state[readMode] = myZero 
        else:
            self.state[readMode] = SerializeMSBTwos(input, K)                        

        self.transitionState, self.transitionLenn = self.GetReadState()

    def SwitchStep(self):
        return self.onSwitchStep 

    def Switch(self):
        #print(f"M2L: Switching mem")
        self.mode = 1 - self.mode

        # Whatever is left unread is discarded from new write mem
        self.state[self.mode] = list()

        self.onSwitchStep = 1        
        self.switchNext = 0
        self.lastVal = 0
                
    def SetSwitchNext(self):        
        self.onSwitchStep = 0
        self.switchNext = 1

    def GotOutput(self) -> int:        
        readMode = 1 - self.mode    
        if len(self.state[readMode]) > 0:
            return 1
        else:
            return 0

    def Output(self) -> int:
        readMode = 1 - self.mode    
        #print(f"Reading with mode {readMode} of length {len(self.state[readMode])}")
        if len(self.state[readMode]) > 0:
            firstVal = self.state[readMode][-1]
            self.lastVal = firstVal
            #lastVal = self.state[readMode][-1]
        else:
            #print(f"WARNING: M2L out of POP!")
            firstVal = self.lastVal

        return firstVal
    
    def Step(self, input) -> None:                
        if self.onSwitchStep == 1:            
            self.onSwitchStep = 0                    
            #self.switchNext = 0
            # update current state estimate
            self.transitionState, self.transitionLenn = self.GetReadState()
        
        # Whatever is left unread is discarded from new write mem        
        self.state[self.mode].append(input)
        
        readMode = 1 - self.mode            
        if len(self.state[readMode]) > 0:  
            self.state[readMode].pop()

        if self.switchNext == 1:
            self.Switch()
        
    def GetTransitionState(self):
        return self.transitionState, self.transitionLenn

    def GetReadState(self):
        readMode = 1 - self.mode
        if len(self.state[readMode]) > 0:            
            state = DeserializeMSBTwos(self.state[readMode])
            lenn = len(self.state[readMode])
        else:
            state = 0
            lenn = 0        
        return state, lenn

    def Print(self, prefix="") -> None:                
        if (len(self.state[0]) > 2):
            #mem0off = DeserializeMSBOffset(self.state[0])
            mem0off = DeserializeMSBTwos(self.state[0])
        else:
            mem0off = 0
                
        if (len(self.state[1]) > 2):
            mem1off = DeserializeMSBTwos(self.state[1])
            #mem1off = DeserializeLSBTwos(self.state[1])
        else:
            mem1off = 0        

        mem0 = [str(el) for el in self.state[0]]            
        mem1 = [str(el) for el in self.state[1]]

        if self.mode == 0:
            print(f"W:{mem0} ({mem0off})")
            print(f"R:{mem1} ({mem1off})")
        else:
            print(f"R:{mem0} ({mem0off})")
            print(f"W:{mem1} ({mem1off})")                        
        
    def Draw(self, ax, x, y, w, h):
        print(f"Drawing M2L to canvas {x}, {y}, {w}, {h}")

        box_h = min(h / 10, w / 2.5)
        gap = box_h * 0.2

        # Calculate column positions
        x_left = x + (w/2 - box_h)/2
        x_left = x + gap 
        x_right = x + 2*gap + box_h

        # Shrink the first green rectangle so that it contains both columns with a small gap on all sides.
        rect_x = x_left - gap
        rect_w = (x_right + box_h + gap) - rect_x
        
        stack_left = self.state[self.mode]
        stack_right = self.state[1-self.mode]
        rect_h = 4 * box_h + 3 * gap + 8

        rect = patches.Rectangle((rect_x, y+gap), rect_w, rect_h, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

        # Draw green line between the stacks
        line_x_mid = x_left + box_h + (x_right - x_left - box_h) / 2
        line = Line2D([line_x_mid, line_x_mid], [y + gap, y + gap + rect_h], color='green', linewidth=1)
        ax.add_line(line)

        # Draw status boxes under the columns
        status_y = y - 12
        
        rect_ss = patches.Rectangle((x_left, status_y), box_h, box_h, linewidth=1, edgecolor='black', facecolor='lightgray' if self.mode else 'white')
        ax.add_patch(rect_ss)
        ax.text(x_left + box_h/2, status_y + box_h/2, f"{self.mode}", ha='center', va='center', fontsize=8)

        rect_sn = patches.Rectangle((x_right, status_y), box_h, box_h, linewidth=1, edgecolor='black', facecolor='lightgray' if self.onSwitchStep else 'white')
        ax.add_patch(rect_sn)
        ax.text(x_right + box_h/2, status_y + box_h/2, f"{self.onSwitchStep}", ha='center', va='center', fontsize=8)

        # Bottom anchor for bottom-up drawing (inside the gap)
        y_bottom_anchor = y + gap + 2

        # Left Stack: self.state[self.mode]
        for j, val in enumerate(stack_left):
            fc = 'lightgray' if val == 1 else 'white'
            r = patches.Rectangle((x_left, y_bottom_anchor + j*box_h), box_h, box_h, linewidth=1, edgecolor='black', facecolor=fc)
            ax.add_patch(r)
            ax.text(x_left + box_h/2, y_bottom_anchor + j*box_h + box_h/2, str(val), ha='center', va='center', fontsize=8)

        # Right Stack: self.state[1-self.mode]
        y_top_anchor = y + rect_h + gap - box_h - 2
        for j, val in enumerate(reversed(stack_right)):
            fc = 'lightgray' if val == 1 else 'white'
            r = patches.Rectangle((x_right, y_top_anchor - j*box_h), box_h, box_h, linewidth=1, edgecolor='black', facecolor=fc)
            ax.add_patch(r)
            ax.text(x_right + box_h/2, y_top_anchor - j*box_h + box_h/2, str(val), ha='center', va='center', fontsize=8)
