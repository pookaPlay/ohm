from OHM import OHM
from DataReader import DataReader
from DataWriter import DataWriter
 
def test_OHM():

    NBitsIn = 4
    NBitsOut = 5
    
    input = [[4, -2], [0, 0], [0, 0], [0, 0]]
    D = len(input[0])
    NSteps = 15

    data = DataReader(input, NBitsIn, NBitsOut)    
    ohm = OHM(D, NBitsIn, NBitsOut)        
    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset(data.Output())        
    output.Reset()

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn(), data.msbIn())
    #ohm.Print("", 1)        

    output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())    
    ohm.Step(data.lsbIn(), data.msbIn())                    

    for bi in range(NSteps):
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        
        ohm.Calc(data.Output(), data.lsbIn(), data.msbIn())
        print(f"OHM: {ohm.Output()}      lsb: {ohm.lsbOut()} msb: {ohm.msbOut()}")
        #ohm.Print("", 1)
        
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        output.Print()

        ohm.Step(data.lsbIn(), data.msbIn())
        
   
    output.PrintAll()
    
    return
        

test_OHM()