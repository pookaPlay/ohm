from DataIO import SerializeMSBTwos, SerializeLSBTwos, DeserializeLSBTwos, DeserializeLSBOffset


class lsb2msb:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0
        readMode = 1 - self.mode    
        self.onSwitchStep = 1
        self.switchNext = 0

    def InitState(self, input, K) -> None:
        #self.state[1-self.mode] = SerializeLSBTwos(input, K)
        self.state[self.mode] = SerializeLSBTwos(input, K)
        #self.state[1 - self.mode] = SerializeLSBTwos(input, K)
        #self.state[1] = SerializeLSBTwos(input, K)
    
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

