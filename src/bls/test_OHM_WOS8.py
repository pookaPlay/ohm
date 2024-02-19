from DataReader import DataReader
from DataWriter import DataWriter
from OHM import OHM


def runTest(input, ptf):
    
    NBitsIn = 8
    NBitsOut = 9
                
    D = len(input[0])
    NSteps = 85

    data = DataReader(input, NBitsIn, NBitsOut)    
    ohm = OHM(D, NBitsIn, NBitsOut, ptf=ptf)        
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
        #ohm.Print("", 1)
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        output.Print()

        ohm.Step(data.lsbIn(), data.msbIn())
        
    output.PrintAll()
    
    return output.Output()

def test_OHM_WOS8():
    ptf = "median"
    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    expected = [2, 0, 1, 1, 2, 0, 1]    
    ret = runTest(input, ptf)    
    
    if ret != expected:
        print(f"Expected: {expected}")
        print(f"Got: {ret}")   
        assert False    


test_OHM_WOS8()
