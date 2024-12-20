from bls.DataReader import DataReader
from bls.LayeredMultiDataWriter import LayeredMultiDataWriter
from bls.OHM_NET import OHM_NET

def test_NET():
    
    param = {
        "debugDone" : 1,
        "flagThresh" : -1,
        "ptf" : "max",
        "nsteps" : 25,        
        "K" : 4,        
        "W" : 3,
        "L" : 3        
    }
    
    input = [6, 7, 5]  
    input = [input.copy() for _ in range(10)]
    
    param["D"] = len(input[0])   
    D = param["D"]
    K = param["K"]
    W = param["W"]
    L = param["L"]

    data = DataReader(input, K, K)    

    ohm = OHM_NET(param)        

    output = LayeredMultiDataWriter(L, W)
        
    data.Reset()        
    ohm.Reset()            

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn())
    
    output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugOut())
    
    ohm.Step()                        

    for bi in range(param["nsteps"]):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()        

        ohm.Calc(data.Output(), data.lsbIn())        

        output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugOut())                               
        ohm.Step()
        
    #output.Print()
    output.BatchProcess()
    output.PrintFinal()
    #ohm.Print(">>>", 1)
    return output.Output()

test_NET()
