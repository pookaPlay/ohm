from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
from bls.OHM_v2 import OHM_v2


def RunNode(input, ptf):
    
    NBitsIn = 8
    NBitsOut = 9
                
    D = len(input[0])
    NSteps = 85

    data = DataReader(input, NBitsIn, NBitsOut)    
    ohm = OHM_v2(D, NBitsIn, NBitsOut, ptf=ptf)        
    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn())
    output.Step(ohm.Output(), ohm.lsbOut())    
    ohm.Step(data.lsbIn())                    

    for bi in range(NSteps):
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        
        ohm.Calc(data.Output(), data.lsbIn())
        print(f"OHM: {ohm.Output()}      lsb: {ohm.lsbOut()}")
        #ohm.Print("", 1)
        output.Step(ohm.Output(), ohm.lsbOut())            
        output.Print()

        ohm.Step(data.lsbIn())
        
    output.PrintAll()
    
    return output.Output()

