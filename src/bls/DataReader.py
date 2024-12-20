from bls.DataIO import SerializeMSBTwos, SerializeLSBTwos, SerializeMSBOffset, SerializeLSBOffset

class DataReader():
    def __init__(self, input=[[5, 7, 6]], NBitsIn=7, NBitsOut=8):
        
        self.NIn = NBitsIn
        self.NOut = NBitsOut
        self.input = input
        self.N = len(input)
        self.D = len(input[0])
        self.bi = 0
        self.ni = 0

        #self.data = [SerializeLSBOffset(input[self.ni][d], self.NIn) for d in range(self.D)]
        self.data = [SerializeLSBTwos(input[self.ni][d], self.NIn) for d in range(self.D)]

        for d in range(self.D):
            print(f"Input: {self.input[self.ni][d]} -> {self.data[d]}")

        self.slice = [self.data[d][self.bi] for d in range(self.D)]        
        self.lsb = [1] * self.D
        self.msb = 0
        print(self.data)


    def Reset(self):
        self.bi = 0
        self.ni = 0
        self.slice = [self.data[d][self.bi] for d in range(self.D)]        
        self.lsb = [1] * self.D
        self.msb = 0
 
    def Print(self, prefix="", verbose=1):                
        print(f"Data Reader: {self.slice} from {self.input[self.ni]} ")
        print(f"Data Reader: lsb({self.lsb})")

    def Output(self):
        return self.slice

    def lsbIn(self):
        return self.lsb

    def Step(self):
        if self.msb == 1:  
            print(f"Serializing next")                      
            self.bi = 0                        
            self.ni = self.ni + 1

            if self.ni == self.N:
                self.ni = 0

            #self.data = [SerializeLSBOffset(self.input[self.ni][d], self.NIn) for d in range(self.D)]
            self.data = [SerializeLSBTwos(self.input[self.ni][d], self.NIn) for d in range(self.D)]
            for d in range(self.D):                
                print(f"Input: {self.input[self.ni][d]} -> {self.data[d]}")                
        else:
            self.bi = self.bi + 1
        
        if self.bi < self.NIn:
            self.slice = [self.data[d][self.bi] for d in range(self.D)]
        elif self.bi < self.NOut:
            # sign extend twos complement
            self.slice = [self.data[d][self.NIn-1] for d in range(self.D)]
        else:            
            self.slice = self.D * [0]

        self.msb = 0
        self.lsb = [0] * self.D

        if self.bi == self.NOut-1:
            self.msb = 1
        if self.bi == 0:
            self.lsb = [1] * self.D
