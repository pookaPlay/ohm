from bls.DataIO import DeserializeLSBTwos, DeserializeLSBOffset

class DataWriter_v2():
    def __init__(self):
        self.result = list()
        self.lsb = list()
        self.lastResult = -1
        self.finalResult = list()

    def Print(self):        
        print(f"LSB   : {self.lsb}")
        print(f"Result: {self.result}")
        
        #print(f"Data Writer: {self.lastResult}")

    def PrintFinal(self):        
        print(f"Data Writer: {self.finalResult}")

    def Output(self):
        return(self.finalResult)

    def Reset(self):
        self.result = list()        
        self.lsb = list()
        self.finalResult = list()


    def Step(self, x, lsb):
        self.result.append(x)
        self.lsb.append(lsb)        
    
    def BatchProcess(self):        
        firstOne = -1
        secondOne = -1
        for i in range(len(self.result)):
            if self.lsb[i] == 1:
                if firstOne == -1:
                    firstOne = i
                else:
                    if secondOne == -1:
                        secondOne = i
                    else:
                        firstOne = secondOne
                        secondOne = i   

                    #result = DeserializeLSBTwos(self.result[firstOne:secondOne])
                    result = DeserializeLSBOffset(self.result[firstOne:secondOne])
                    print(f"Got Result: {result} from length {len(self.result[firstOne:secondOne])}")
                    self.finalResult.append(result)
                
                    
        

        
        

