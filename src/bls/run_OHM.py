from DataReader import DataReader
from DataWriter import DataWriter
from OhmNet import OhmNet

def RunOhmNet():
    
    verbose = 1

    
    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    
    NBitsIn = 4
    NBitsOut = 5

    D = len(input[0])
    NSteps = 45

    data = DataReader(input, NBitsIn, NBitsOut)    
    ohm = OhmNet(D, NBitsIn, NBitsOut)        
    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn(), data.msbIn())
    output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())    
    ohm.Step(data.lsbIn(), data.msbIn())                    

    for bi in range(NSteps):
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        
        ohm.Calc(data.Output(), data.lsbIn(), data.msbIn())
        
        print(f"OHM: {ohm.Output()} lsb: {ohm.lsbOut()} msb: {ohm.msbOut()}  done: {ohm.done()}")
        ohm.Print("", verbose)
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        output.Print()

        ohm.Step(data.lsbIn(), data.msbIn())
        
    output.PrintAll()
    
    return output.Output()


RunOhmNet()
