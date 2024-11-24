from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
from bls.OHM import OHM

def test_NET():
    
    input = [6, 1, 2]  # produces alternating 2 bit outputs
    #input = [6, 7, 5]  # works with max    
    input = [input.copy() for _ in range(10)]

    param = {
        "ptf" : "max",
        "nsteps" : 16,        
        "K" : 4,
        "D" : 2,
    }

    param["D"] = len(input[0])   
    D = param["D"]
    K = param["K"]

    data = DataReader(input, K, K)    

    ohm = OHM(param)        

    output = DataWriter()    
    
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

    for bi in range(param["nsteps"]):
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

test_NET()
