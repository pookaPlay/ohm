from bls.DataWriter import DataWriter

class MultiDataWriter():
    def __init__(self, N):
        self.writers = [DataWriter() for _ in range(N)]

    def Print(self):        
        [writer.Print() for writer in self.writers]        

    def PrintFinal(self):        
        [writer.PrintFinal() for writer in self.writers]        

    def Output(self):
        result = [writer.Output() for writer in self.writers]
        return(result)

    def Reset(self):
        [writer.Reset() for writer in self.writers]

    def Step(self, x, lsb, debugTicksTaken):
        assert(len(x) == len(self.writers))
        for i in range(len(x)):
            self.writers[i].Step(x[i], lsb[i], debugTicksTaken[i])        
    
    def BatchProcess(self):      
        [writer.BatchProcess() for writer in self.writers]
                
                    
        

        
        

