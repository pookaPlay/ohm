
class msb2lsb_v2:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0                
        self.onSwitchStep = 0
        self.switchNext = 0

    def SwitchStep(self):
        return self.onSwitchStep 

    def Switch(self):
        self.mode = 1 - self.mode
        self.onSwitchStep = 1        
        self.switchNext = 0
                
    def SetSwitchNext(self):        
        self.onSwitchStep = 0
        self.switchNext = 1

    def Output(self) -> int:
        readMode = 1 - self.mode    
        #print(f"Reading with mode {readMode} of length {len(self.state[readMode])}")
        if len(self.state[readMode]) > 0:
            firstVal = self.state[readMode][-1]
        else:
            print(f"WARNING: M2L out of POP!")
            firstVal = 0

        return firstVal
    
    def Step(self, input) -> None:        
        
        self.state[self.mode].append(input)        
        
        if len(self.state[1 - self.mode]) > 1:  # hold last one
            self.state[1 - self.mode].pop()

        if self.onSwitchStep == 1:
            self.onSwitchStep = 0

        if self.switchNext == 1:
            self.Switch()

    def Print(self, prefix="") -> None:        
        temps = prefix + "M2L: "

        mem0 = [str(el) for el in self.state[0]]            
        mem1 = [str(el) for el in self.state[1]]
        #print(temps + "]")

        if self.mode == 0:
            print(f"W:{mem0}")
            print(f"R:{mem1}")
        else:
            print(f"R:{mem0}")
            print(f"W:{mem1}")
            
            
        

