from bls.DataIO import DeserializeLSBTwos, DeserializeLSBOffset

PTF_DELAY = 1

class msb2lsb:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0                
        self.onSwitchStep = 0
        self.switchNext = 0
        self.lastVal = 0
        self.ptfCount = 0

    def SwitchStep(self):
        return self.onSwitchStep 

    def Switch(self):
        # Right before switch 
        #if len(self.state[self.mode]) > 0:
        #    self.state[self.mode][-1] = 1 - self.state[self.mode][-1]

        print(f"M2L: Switching mem")        
        self.mode = 1 - self.mode        
        self.onSwitchStep = 1        
        self.switchNext = 0
        self.ptfCount = 0
        # clear the write mem
        #self.state[self.mode].clear()

                
    def SetSwitchNext(self):        
        self.onSwitchStep = 0
        self.switchNext = 1

    def Output(self) -> int:
        readMode = 1 - self.mode    
        #print(f"Reading with mode {readMode} of length {len(self.state[readMode])}")
        if len(self.state[readMode]) > 0:            
            firstVal = self.state[readMode][-1]            
            self.lastVal = firstVal                        
        else:
            firstVal = self.lastVal

        # elif len(self.state[readMode]) == 1:
        #     firstVal = 1 - self.state[readMode][-1]
        #     self.lastVal = firstVal

        #if self.onSwitchStep == 1:
        #    # flip MSB
        #    #firstVal = 1 - firstVal
        #    #if len(self.state[readMode]) > 0:
        #    #    self.state[readMode][-1] = firstVal

        return firstVal
    
    def Step(self, input, validOut) -> None:        
        
        self.ptfCount = self.ptfCount + 1

        if self.onSwitchStep == 1:
            self.onSwitchStep = 0            
            self.inputValue = 1-input                                   
        else:
            self.inputValue = input                       
        
        if validOut == 1:
            self.state[self.mode].append(self.inputValue)

        if len(self.state[1 - self.mode]) > 1:  # hold last one
            self.state[1 - self.mode].pop()                    

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
            
            
