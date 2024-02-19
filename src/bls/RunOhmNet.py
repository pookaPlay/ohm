from DataReader import DataReader
from DataWriter import DataWriter
from OhmNet import OhmNet1

def RunOhmNet():
    
    ptf = "min"
    NBitsIn = 4
    NBitsOut = 5
                
    D = len(input[0])
    NSteps = 45

    data = DataReader(input, NBitsIn, NBitsOut)    
    ohm = OHMNet1(D, NBitsIn, NBitsOut, ptf=ptf)        
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
        print(f"OHM: {ohm.Output()}      lsb: {ohm.lsbOut()} msb: {ohm.msbOut()}")
        ohm.Print()
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        output.Print()

        ohm.Step(data.lsbIn(), data.msbIn())
        
    output.PrintAll()
    
    return output.Output()

