from bls.DataReader import DataReader
from bls.LayeredMultiDataWriter import LayeredMultiDataWriter
from bls.OHM_NET import OHM_NET
import random

random.seed(0)

def test_NET():

    K = 16
    D = 16
    L = 2
    W = 16
    max_value = 2 ** (K - 1)

    param = {
        "debugDone" : 0,
        "debugTree" : 1,
        "flagThresh" : -1,
        "ptf" : "median",
        "nsteps" : K*4,        
        "K" : K,
        "D" : D,
        "L" : L,
        "W" : W,
    }

    
    input = [random.randint(1, max_value) for _ in range(D)]
    #input = [6, 7, 5]     
    #input = [1, 2, 4]  
    print(f"Input: {input}")
    
    input = [input.copy() for _ in range(10)]
    D = len(input[0])
    param["D"] = D


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
