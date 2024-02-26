##########################
## Bit Serial Stack Filter

from PTF import PTF

class STACK:
    
    def __init__(self, D=2, ptf="") -> None:

        self.D = D        

        self.flags = list(self.D * [0])                        
        # This is a hack - i dont think I actually need to store latchInput
        self.latchInput = list(self.D * [0])

        self.pbf = PTF(self.D)                
        # Some presets for debugging
        if ptf == "min":
            self.pbf.SetMin()
        elif ptf == "max":          
            self.pbf.SetMax()                           
        else:       
            self.pbf.SetMedian()
        
        self.done = 0
        
    def Reset(self) -> None:
        
        self.flags = list(self.D * [0])                        
        self.latchInput = list(self.D * [0])
        self.pbf.Reset()                
        self.done = 0        
                
    def Output(self) -> int:
        return self.pbf.Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x, msb=0) -> None:        

        # Get the inputs for the PBF
        #inputs = [x.Output() for i in range(len(x))]
        inputs = x.copy()
        # Negate if msb to convert to offset
        if msb == 1:            
            inputs = [1-x for x in inputs]
            self.flags = list(self.D * [0])
            self.latchInput = list(self.D * [0])
            self.done = 0
            # print(f"Negated inputs: {inputs}")
        else:
            for i in range(self.D):    
                if self.flags[i] == 1:
                    inputs[i] = self.latchInput[i]

        self.pbf.Calc(inputs)

        for i in range(self.D):
            if self.flags[i] == 0:
                if inputs[i] != self.pbf.Output():
                    self.flags[i] = 1
                    self.latchInput[i] = inputs[i]
                                        
        if (sum(self.flags) == (self.D-1)):            
            self.done = 1            
        
    # State stuff goes here
    def Step(self):        
        pass                

    def Print(self, prefix="", showInput=1) -> None:
        print(f"{prefix}STACK: {self.D} bits")
        self.pbf.Print(f"{prefix} ")

