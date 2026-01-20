from DataIO import SerializeMSBTwos, SerializeLSBTwos, SerializeMSBOffset, DeserializeLSBTwos, DeserializeMSBTwos, DeserializeLSBOffset, DeserializeMSBOffset


class msb2lsb:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0                
        self.onSwitchStep = 1
        self.switchNext = 0

    def InitState(self, input, K) -> None:
        self.state[1-self.mode] = SerializeMSBTwos(input, K)        

    def SwitchStep(self):
        return self.onSwitchStep 

    def Switch(self):
        print(f"M2L: Switching mem")
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
        
        if self.onSwitchStep == 1:
            self.onSwitchStep = 0                    
            
        if len(self.state[1 - self.mode]) > 0:  # hold last one
            self.state[1 - self.mode].pop()                    

        if self.switchNext == 1:
            self.Switch()
        
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
        

