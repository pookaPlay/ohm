###################
## Bit Serial Adder

import matplotlib.patches as patches

class ADD:
    def __init__(self) -> None:        
        self.sum = 0
        self.cin = 0
        self.a = 0
        self.b = 0
        self.cout = 0

    def Print(self, prefix="", verbose=1) -> None:
        if verbose > 0:
            print(prefix + f"ADD : {self.sum}, {self.cout} from {self.a}, {self.b} and {self.cin}")
    
    def Output(self) -> int:
        return self.sum

    def Reset(self) -> None:            
        self.sum = 0
        self.cin = 0
        self.cout = 0        

    def Calc(self, a, b, lsb=0) -> None:                
        if lsb == 1:
            self.cin = 0

        self.a = a
        self.b = b
        self.sum  = (a+b+self.cin) % 2
        self.cout = 1 if (a+b+self.cin) > 1 else 0

    def Step(self) -> None:                
        self.cin = self.cout
        #print(f"  After step ADD : {self.sum} and {self.cin}")

    def Draw(self, ax, x, y, w, h):
        #rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', facecolor='none')
        #ax.add_patch(rect)
        #ax.text(x + w/2, y + h/2, f"ADD\nS:{self.sum}\nC:{self.cout}", ha='center', va='center', fontsize=8)
        
        #rect_ss = patches.Rectangle((x_left, status_y), box_h, box_h, linewidth=1, edgecolor='black', facecolor='lightgray' if self.mode else 'white')        

        state_x = x
        state_y = y
        latch_w = 8
        latch_h = 12
        rect_latch = patches.Rectangle((state_x, state_y), latch_w, latch_h, linewidth=1, edgecolor='black', facecolor='lightgray' if self.sum else 'white')
        ax.add_patch(rect_latch)
        ax.text(state_x + latch_w/2, state_y + latch_h/2, str(self.sum), ha='center', va='center', fontsize=8)
        carry_x = x
        carry_y = y+16
        rect_latch = patches.Rectangle((carry_x, carry_y), latch_w, latch_h, linewidth=1, edgecolor='black', facecolor='lightgray' if self.cout else 'white')
        ax.add_patch(rect_latch)
        ax.text(carry_x + latch_w/2, carry_y + latch_h/2, str(self.cout), ha='center', va='center', fontsize=8)
