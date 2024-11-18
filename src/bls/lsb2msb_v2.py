from bls.DataIO import SerializeMSBTwos, SerializeLSBTwos

class lsb2msb_v2:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self, inputId = -1) -> None:
        self.state = [list(), list()]        
        self.mode = 0
        readMode = 1 - self.mode    
        if inputId > -1: 
            #initId = SerializeMSBTwos(inputId, 4)
            initId = SerializeLSBTwos(inputId, 4)
            print(f"Initializing L2M with {initId}")
            self.state[readMode] = initId

    def Output(self) -> int:
        readMode = 1 - self.mode    
        
        if len(self.state[readMode]) > 0:
            firstVal = self.state[readMode][-1]
        else:
            #print(f"WARNING: L2M out of POP!")
            firstVal = 0

        return firstVal
    
    def Step(self, input) -> None:
        #print(f"L2M write at {self.wi} : {input}")
        self.state[self.mode].append(input)        
        
        if len(self.state[1 - self.mode]) > 0:
            self.state[1 - self.mode].pop()
        
        
    def Switch(self):
        self.mode = 1 - self.mode

    def Print(self, prefix="") -> None:        
        temps = prefix + "L2M: "

        mem0 = [str(el) for el in self.state[0]]            
        mem1 = [str(el) for el in self.state[1]]
        #print(temps + "]")
        if self.mode == 0:
            print(f"W:{mem0}")
            print(f"R:{mem1}")
        else:
            print(f"R:{mem0}")
            print(f"W:{mem1}")
            
            
        

