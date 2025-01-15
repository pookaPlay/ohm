from bls.DataReader_v2 import DataReader_v2
from bls.DataWriter_v2 import DataWriter_v2
from bls.OHM_v4 import OHM_v4
from bls.lsbSource import lsbSource

def RunNode(input, ptf, NBitsIn = 4, NBitsOut = 4, NSteps = 120):

    K = 4
    D = len(input[0])   

    data = DataReader_v2(input, NBitsIn, NBitsOut)    

    wZero = K * [0]
    wOne = wZero.copy()
    wOne[0] = 1        
    print(f"Defaults: {wZero} and {wOne}")

    wp = [lsbSource(K, wZero) for _ in range(D)]        
    wn = [lsbSource(K, wOne) for _ in range(D)]        

    ohm = OHM_v4(D, K, K, ptf=ptf)        

    output = DataWriter_v2()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()
    [wpi.Reset() for wpi in wp]
    [wni.Reset() for wni in wn]

    print(f"== {0} ============================")
    data.Print()

    wpin = [wpi.Output() for wpi in wp]
    wnin = [wni.Output() for wni in wn]

    ohm.Calc(data.Output(), wpin, wnin, data.lsbIn())
    #print(f"--- OUT: {ohm.Output()}      LSB: {ohm.lsbOut()}")
    #ohm.Print("", 1)

    output.Step(ohm.Output(), ohm.lsbOut())    
    
    
    ohm.Step()                        

    for bi in range(NSteps):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
                
        [wpi.Step() for wpi in wp]
        [wni.Step() for wni in wn]

        wpin = [wpi.Output() for wpi in wp]
        wnin = [wni.Output() for wni in wn]

        ohm.Calc(data.Output(), wpin, wnin, data.lsbIn())        
        #ohm.Print("", 1)        
        #print(f"--- OUT: {ohm.Output()} LSB: {ohm.lsbOut()}")
        output.Step(ohm.Output(), ohm.lsbOut())            
                
        ohm.Step()
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
    
    return output.Output()

