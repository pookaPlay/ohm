from DataIO import SerializeMSBTwos, SerializeLSBTwos, SerializeMSBOffset, DeserializeLSBTwos, DeserializeMSBTwos, DeserializeLSBOffset, DeserializeMSBOffset

class lsbSource:    
    
    def __init__(self) -> None:             
        self.state = list()
        self.K = 0
        self.ri = 0
        self.lastVal = 0
        self.lsb = 0
    
    def InitState(self, input, K) -> None:
        self.input = input
        self.K = K        
        self.Reset()    
            
    def ZeroState(self, K) -> None:        
        self.InitState(0, K)

    def OneState(self, K) -> None:        
        self.InitState(1, K)

    def Reset(self) -> None:
        self.lastVal = 0
        self.state = SerializeMSBTwos(self.input, self.K)        
        self.lsb = 1

    def Print(self, prefix="") -> None:                
        if (len(self.state) > 0):
            #mem0off = DeserializeMSBOffset(self.state[0])
            mem1off = DeserializeMSBTwos(self.state)
        else:
            mem1off = 0                        
        
        mem1 = [str(el) for el in self.state]        
        print(f"R:{mem1} ({mem1off})")

    def LsbOut(self) -> int:
        return self.lsb

    def Output(self) -> int:        
        if len(self.state) > 0:
            firstVal = self.state[-1]
            self.lastVal = firstVal
        else:            
            #print(f"WARNING: L2S out of POP!")            
            firstVal = self.lastVal

        return firstVal

    def Step(self) -> None:       
        
        if self.lsb == 1:
            self.lsb = 0

        if len(self.state) > 0:  
            self.state.pop()

        if len(self.state) == 0:
            self.Reset()            
                
