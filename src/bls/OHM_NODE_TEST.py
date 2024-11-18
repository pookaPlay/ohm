from bls.DataReader_v2 import DataReader_v2
from bls.DataWriter_v2 import DataWriter_v2
from bls.OHM_v2 import OHM_v2


def RunNode(input, ptf, NBitsIn = 4, NBitsOut = 4, NSteps = 120):

    D = len(input[0])   

    data = DataReader_v2(input, NBitsIn, NBitsOut)    
    ohm = OHM_v2(D, NBitsIn, NBitsOut, ptf=ptf)        
    output = DataWriter_v2()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn())
    print(f"--- OUT: {ohm.Output()}      LSB: {ohm.lsbOut()}")
    #ohm.Print("", 1)

    output.Step(ohm.Output(), ohm.lsbOut())    
    
    print(f"--- Stepping OHM with data lsb {data.lsbIn()}")
    ohm.Step(data.lsbIn())                    

    for bi in range(NSteps):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        
        ohm.Calc(data.Output(), data.lsbIn())        
        #ohm.Print("", 1)
        
        print(f"--- OUT: {ohm.Output()} LSB: {ohm.lsbOut()}")
        output.Step(ohm.Output(), ohm.lsbOut())            
        
        print(f"--- OHM STEP LSB IN: {data.lsbIn()}")
        ohm.Step(data.lsbIn())
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
    
    return output.Output()

