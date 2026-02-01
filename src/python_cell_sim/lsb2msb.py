from DataIO import SerializeMSBTwos, SerializeLSBTwos, DeserializeLSBTwos, DeserializeLSBOffset
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class lsb2msb:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0
        readMode = 1 - self.mode    
        self.onSwitchStep = 1
        self.switchNext = 0

    def InitState(self, input, K, initMode = 1) -> None:            
        if initMode == 0:            
            myZero = [0] * K
            self.state[self.mode] = myZero 
            self.state[1-self.mode] = myZero 
        elif initMode == 1:
            self.state[self.mode] = SerializeLSBTwos(input, K)                
        elif initMode == 2:
            self.state[self.mode] = SerializeLSBTwos(input, K)
            self.state[1-self.mode] = SerializeLSBTwos(input, K)
    
    def GotOutput(self) -> int:
        readMode = 1 - self.mode    
        if len(self.state[readMode]) > 0:
            return 1
        else:
            return 0

    def Output(self) -> int:
        readMode = 1 - self.mode    
        
        if len(self.state[readMode]) > 0:
            firstVal = self.state[readMode][-1]
        else:            
            #print(f"WARNING: L2M out of POP!")            
            firstVal = 0

        # Twos complement negation here
        if self.onSwitchStep == 1:
            # negate msb
            # print(f"  - Negating MSB")
            firstVal = 1 - firstVal

        return firstVal
    
    def Switch(self):
        self.mode = 1 - self.mode
        self.onSwitchStep = 1
        self.switchNext = 0
        # Whatever is left unread is discarded from new write mem
        self.state[self.mode] = list()

    def SetSwitchNext(self):        
        self.onSwitchStep = 0
        self.switchNext = 1

    def Step(self, input) -> None:
        readMode = 1 - self.mode    
        #print(f"L2M write at {self.wi} : {input}")
        self.state[self.mode].append(input)        
        
        if len(self.state[readMode]) > 0:            
            self.state[readMode].pop()
        
        if self.onSwitchStep == 1:
            self.onSwitchStep = 0
        
        if self.switchNext == 1:
            self.Switch()

        
    def Print(self, prefix="") -> None:                
        if (len(self.state[0]) > 2):
            #mem0off = DeserializeLSBOffset(self.state[0])
            mem0off = DeserializeLSBTwos(self.state[0])
        else:
            mem0off = 0
                
        if (len(self.state[1]) > 2):
            #mem1off = DeserializeLSBOffset(self.state[1])
            mem1off = DeserializeLSBTwos(self.state[1])
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
        print(f"Drawing L2M to canvas {x}, {y}, {w}, {h}")

        box_h = min(h / 10, w / 2.5)
        gap = box_h * 0.2

        # Calculate column positions
        x_left = x + (w/2 - box_h)/2
        x_right = x + w/2 + (w/2 - box_h)/2

        # Shrink the first green rectangle so that it contains both columns with a small gap on all sides.
        rect_x = x_left - gap
        rect_w = (x_right + box_h + gap) - rect_x
        
        stack_left = self.state[self.mode]
        stack_right = self.state[1-self.mode]
        rect_h = max(len(stack_left), len(stack_right)) * box_h + 3 * gap

        rect = patches.Rectangle((rect_x, y+gap), rect_w, rect_h, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

        # Top anchor for top-down drawing (inside the gap)
        y_top_anchor = y + gap + rect_h - gap

        # Left Stack: self.state[self.mode]
        for j, val in enumerate(stack_left):
            fc = 'lightgray' if val == 1 else 'white'
            r = patches.Rectangle((x_left, y_top_anchor - (j+1)*box_h), box_h, box_h, linewidth=1, edgecolor='black', facecolor=fc)
            ax.add_patch(r)
            ax.text(x_left + box_h/2, y_top_anchor - (j+1)*box_h + box_h/2, str(val), ha='center', va='center', fontsize=8)

        # Right Stack: self.state[1-self.mode]
        for j, val in enumerate(stack_right):
            fc = 'lightgray' if val == 1 else 'white'
            r = patches.Rectangle((x_right, y_top_anchor - (j+1)*box_h), box_h, box_h, linewidth=1, edgecolor='black', facecolor=fc)
            ax.add_patch(r)
            ax.text(x_right + box_h/2, y_top_anchor - (j+1)*box_h + box_h/2, str(val), ha='center', va='center', fontsize=8)


if  __name__ == "__main__":
    
    # test draw method
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    l2m = lsb2msb()
    l2m.InitState(7, 8, 2)
    
    l2m.Draw(ax, 0, 0, 10, 10)    
    plt.show()  
