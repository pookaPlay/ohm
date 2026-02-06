import math
import matplotlib.patches as patches

class PTF:
    def __init__(self, D=3) -> None:
        self.D = D
        self.weights = list(self.D * [1])
        self.threshold = self.D/2
        #self.threshold = self.D
        self.y = 0
        self.lastx = list(self.D * [0])
        self.strFunc = "med"
            
    def SetSort(self, index, outOf) -> None:        
        self.weights = list(self.D * [1])
        self.threshold = index+1
        if self.threshold > self.D:
            self.threshold = self.D
        self.strFunc = "sort"
        print(f"sort init: {self.threshold}")

    def SetMin(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D 
        self.strFunc = "min"

    def SetMax(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = 1
        self.strFunc = "max"

    def SetMedian(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D/2
        self.strFunc = "med"

    def Output(self) -> int:
        return self.y

    def Calc(self, x) -> None:        
        self.lastx = x.copy()
        temp = sum([self.weights[i] for i in range(self.D) if x[i] == 1])
        self.y = 1 if temp >= self.threshold else 0        

    def Print(self, prefix=""):
        #print(f"{prefix} PTF({self.strFunc}): {self.lastx} -> {self.y}")
        print(f"{prefix} PTF({self.weights}): {self.threshold}")

    def Step(self, x) -> None:
        self.Calc(x)
        pass

    def Reset(self) -> None:
        self.y = 0

    def Draw(self, ax, x, y, w, h):
        #rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='orange', facecolor='none')
        #ax.add_patch(rect)
        #ax.text(x + w/2, y + h/2, f"PTF\n{self.strFunc}\nY:{self.y}", ha='center', va='center', fontsize=8)

        state_x = x
        state_y = y
        latch_w = 8
        latch_h = 12
        rect_latch = patches.Rectangle((state_x, state_y), latch_w, latch_h, linewidth=1, edgecolor='black', facecolor='lightgray' if self.y else 'white')
        ax.add_patch(rect_latch)
        ax.text(state_x + latch_w/2, state_y + latch_h/2, str(self.y), ha='center', va='center', fontsize=8)

