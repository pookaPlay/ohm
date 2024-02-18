from CCSRL import CCSRL
from ADD import ADD
from PBF import PBF
from SRL import SRL
from msb2lsb import msb2lsb
from lsb2msb import lsb2msb

class OHM:
    
    def __init__(self, D=2, Nin = 4, Nout = 5) -> None:
        ## Nin is the stored precision of weights
        ## NOut is the sign extened precision
        ## Input should be two's complement with NOut bits

        self.d = D
        self.d2 = D*2        
        self.Nin = Nin
        self.Nout = Nout
        # default weights
        self.zeros = list(self.Nin * [0])
        self.one = self.zeros.copy()
        self.one[self.Nin-1] = 1

        self.wp = [CCSRL(self.Nin, self.Nout) for _ in range(self.d)]
        self.wn = [CCSRL(self.Nin, self.Nout) for _ in range(self.d)]        
        self.addp = [ADD() for _ in range(self.d)]
        self.addn = [ADD() for _ in range(self.d)]
        
        self.lsb2msb = [lsb2msb(self.Nout) for _ in range(self.d2)]
        
        ## NBits+1 is the input precision for the PBF
        ## Need to flip NBits+1 MSB to convert twos' to offset binary

        self.msbAtPBF = SRL(1)
        self.flags = list(self.d2 * [0])                        
        self.pbf = PBF(self.d2)                
        
        self.msb2lsb = msb2lsb(self.Nout)
        
    def Reset(self, x) -> None:
        
        for i in range(self.d):

            self.wp[i].Reset(self.zeros)
            self.addp[i].Reset(x[i], self.wp[i].Output())

            self.wn[i].Reset(self.one)
            self.addn[i].Reset(x[i], self.wn[i].Output())
                            
            self.lsb2msb[i].Reset()
            self.lsb2msb[i+self.d].Reset()
            
                    
        self.msbAtPBF.Reset()
        self.flags = list(self.d2 * [0])                        

        self.pbf.Reset([self.lsb2msb[i].Output() for i in range(self.d2)])
        self.msb2lsb.Reset()        
        
        self.Calc(x)


    def Output(self) -> int:
        return self.msb2lsb.Output()
        
    # Combinatorial stuff goes here
    def Calc(self, x) -> None:        
        
        nx = [1-x[i] for i in range(len(x))]

        for i in range(self.d):
            self.addp[i].Calc(x[i], self.wp[i].Output()) 
            self.addn[i].Calc(nx[i], self.wn[i].Output())

        inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        # Negate if msb to convert to offset
        if self.msbAtPBF.Output() == 1:            
            inputs = [1-x for x in inputs]
            # print(f"Negated inputs: {inputs}")

        self.pbf.Calc(inputs)
        self.flags = [1 if inputs[i] != self.pbf.Output() else 0 for i in range(self.d2)]
        
    # State stuff goes here
    def Step(self, isMsb) -> None:        

        self.msb2lsb.Step(self.pbf.Output())
        
        self.msbAtPBF.Step(isMsb)

        for i in range(self.d):
            self.lsb2msb[i].Step(self.addp[i].Output(), self.flags[i])
            self.lsb2msb[i+self.d].Step(self.addn[i].Output(), self.flags[i+self.d])
            
            self.addp[i].Step()  
            self.addn[i].Step()

            self.wp[i].Step()
            self.wn[i].Step()

    def Print(self, prefix="", showInput=1) -> None:
        #print(f"==============================")
        #print(f"OHM: {self.d2} inputs")

        if showInput:            
            print(f" +ve ------------------------")
            for i in range(self.d):                
                prefix = f"   x{i}-"
                #self.wp[i].Print(prefix)
                #self.addp[i].Print(prefix)
                self.lsb2msb[i].Print(prefix)                        
            
            print(f" -ve ------------------------")
            for i in range(self.d):                                
                prefix = f"   x{i}-"                                                
                #self.wn[i].Print(prefix)
                #self.addn[i].Print(prefix)
                self.lsb2msb[i+self.d].Print(prefix)

        prefix = "  "
        #inputs = [self.lsb2msb[i].Output() for i in range(self.d2)]
        print(f" =Output =====")
        #self.msbAtPBF.Print()        
        self.pbf.Print(" ")
        #print(f"  PBF: {str(inputs)} -> {self.pbf.Output()}")        
        self.msb2lsb.Print(prefix)

