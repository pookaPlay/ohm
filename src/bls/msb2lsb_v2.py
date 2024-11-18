
class msb2lsb_v2:    
    
    def __init__(self) -> None:             
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(), list()]        
        self.mode = 0                

    def Output(self) -> int:
        readMode = 1 - self.mode    
        #print(f"Reading with mode {readMode} of length {len(self.state[readMode])}")
        if len(self.state[readMode]) > 0:
            firstVal = self.state[readMode][-1]
            # if len(self.state[readMode]) == 1:
            #     firstVal = 1 - self.state[readMode][-1]
            #     print(f"###############################################")
            #     print(f"###############################################")
            #     print(f"NOTE: Returning MSB negated")
            #     print(f"###############################################")
            #     print(f"###############################################")
        else:
            print(f"WARNING: M2L out of POP!")
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
            
            
        

