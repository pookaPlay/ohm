from DataIO import SerializeMSBTwos

class SingleDataReader():
    def __init__(self, input=[5, 7, 6], NBitsIn=8, NBitsOut=8):
        
        if isinstance(input, int):
            self.input = [input]
        else:
            self.input = input.copy()

        self.N = len(self.input)
        #print(f"Single: {self.input} has {self.N}")
        self.NIn = NBitsIn
        self.NOut = NBitsOut                
        self.bi = 0
        self.ni = 0
        
        self.data = SerializeMSBTwos(self.input[self.ni], self.NIn)
        self.data.reverse()

        self.slice = self.data[self.bi]
        self.lsb = 1
        self.msb = 0


    def Reset(self):
        self.bi = 0
        self.ni = 0
        self.slice = self.data[self.bi]
        self.lsb = 1
        self.msb = 0
 
    def Print(self, prefix="", verbose=1):                
        print(f"Single Reader: {self.slice} from {self.input[self.ni]} ")
        print(f"Single Reader: lsb({self.lsb}) msb({self.msb})")

    def Output(self):
        return self.slice

    def lsbIn(self):
        return self.lsb

    def msbIn(self):
        return self.msb

    def Step(self):
        
        if self.msb == 1:  
            #print(f"Serializing next")                      
            self.bi = 0                        
            self.ni = self.ni + 1

            if self.ni == len(self.input):
                self.ni = 0
            self.data = SerializeMSBTwos(self.input[self.ni], self.NIn)
            self.data.reverse()                
        else:
            self.bi = self.bi + 1
        
        if self.bi < self.NIn:
            self.slice = self.data[self.bi]
        elif self.bi < self.NOut:
            # sign extend twos complement
            self.slice = self.data[self.NIn-1]
        else:            
            self.slice = [0]

        self.msb = 0
        self.lsb = 0

        if self.bi == self.NOut-1:
            self.msb = 1
        if self.bi == 0:
            self.lsb = 1
