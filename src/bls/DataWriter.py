from DataIO import SerializeMSBTwos

class DataWriter():
    def __init__(self, NBitsIn=7):
        self.bi = 0
        self.NIn = NBitsIn        

    def Reset(self):
        pass
    def Print(self):        
        pass
    def Output(self):
        return self.slice


    def Step(self):
        self.bi = self.bi + 1
        
        if self.bi < self.NIn:
            pass
