from bls.DataIO import DeserializeLSBTwos, DeserializeLSBOffset

class DataWriter():
    def __init__(self):
        self.result = list()
        self.lsb = list()
        self.ticksTaken = list()
        self.lastResult = -1
        self.finalResult = list()
        self.finalLength = list()

    def Print(self):        
        print(f"LSB   : {self.lsb}")
        print(f"Result: {self.result}")
        print(f"Ticks : {self.ticksTaken}")
        #print(f"Data Writer: {self.lastResult}")

    def PrintFinal(self):                
        #temps = f"Data Writer: "
        #for result, length in zip(self.finalResult, self.finalLength):
        #    temps += f"{result} ({length}), "
        print(f"Data Writer: {self.finalResult}")
        print(f"Data Length: {self.finalLength}")
        print(f"Ticks Taken: {self.ticksTaken}")
        

    def Output(self):
        return(self.finalResult)

    def Reset(self):
        self.result = list()        
        self.lsb = list()
        self.finalResult = list()


    def Step(self, x, lsb, ticksTaken):
        self.result.append(x)
        self.lsb.append(lsb)        
        self.ticksTaken.append(ticksTaken)
    
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

                    result = DeserializeLSBTwos(self.result[firstOne:secondOne])
                    resultLength = len(self.result[firstOne:secondOne])
                    #result = DeserializeLSBOffset(self.result[firstOne:secondOne])
                    #print(f"Got Result: {result} from length {resultLength}")
                    self.finalResult.append(result)
                    self.finalLength.append(resultLength)
                
                    
        

        
        

