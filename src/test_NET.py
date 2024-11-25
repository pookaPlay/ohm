from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
from bls.OHM_NET import OHM_NET

def test_NET():
    
    param = {
        "ptf" : "max",
        "nsteps" : 16,        
        "K" : 4,        
        "W" : 3,
        "L" : 1        
    }

    input = [6, 1, 2]  # produces alternating 2 bit outputs
    input = [6, 7, 5]  # works with max    
    input = [input.copy() for _ in range(10)]
    
    param["D"] = len(input[0])   
    D = param["D"]
    K = param["K"]

    data = DataReader(input, K, K)    

    ohm = OHM_NET(param)        

    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn())

    output.Step(ohm.Output(), ohm.lsbOut())            
    
    ohm.Step()                        

    for bi in range(param["nsteps"]):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()        

        ohm.Calc(data.Output(), data.lsbIn())        

        output.Step(ohm.Output(), ohm.lsbOut())                           
    
        ohm.Step()
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
    #ohm.Print(">>>", 1)
    return output.Output()

test_NET()
