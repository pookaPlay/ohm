from bls.DataIO import SerializeMSBTwos, SerializeLSBTwos
from bls.DataIO import DeserializeLSBTwos, DeserializeLSBOffset

class lsb2msb:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0        
        self.onSwitchStep = 0
        self.switchNext = 0

    def Output(self) -> int:
        readMode = 1 - self.mode    
        
        if len(self.state[readMode]) > 0:
            firstVal = self.state[readMode][-1]
        else:
            #print(f"WARNING: L2M out of POP!")
            firstVal = 0

        return firstVal
    
    def Switch(self):
        self.mode = 1 - self.mode
        self.onSwitchStep = 1
        self.switchNext = 0

    def SetSwitchNext(self):        
        self.onSwitchStep = 0
        self.switchNext = 1

    def Step(self, input, flag = 0) -> None:
        #print(f"L2M write at {self.wi} : {input}")
        self.state[self.mode].append(input)        
        
        if len(self.state[1 - self.mode]) > 0:
            if flag == 0:
                self.state[1 - self.mode].pop()
        
        if self.onSwitchStep == 1:
            self.onSwitchStep = 0
        
        if self.switchNext == 1:
            self.Switch()
        
    def Print(self, prefix="") -> None:                
        mem0off = DeserializeLSBOffset(self.state[0])
        #mem0twos = DeserializeLSBTwos(mem0)
        mem1off = DeserializeLSBOffset(self.state[1])
        #mem1twos = DeserializeLSBTwos(mem1)

        mem0 = [str(el) for el in self.state[0]]            
        mem1 = [str(el) for el in self.state[1]]

        if self.mode == 0:
            print(f"W:{mem0} (off: {mem0off})")
            print(f"R:{mem1} (off: {mem1off})")
        else:
            print(f"R:{mem0} (off: {mem0off})")
            print(f"W:{mem1} (off: {mem1off})")
            
            
        

