from DataReader import DataReader
from DataWriter import DataWriter
from OHM import OHM


def runTest(ptf, offsets=list()):
    NBitsIn = 4
    NBitsOut = 5
    
    input = [[7, 2, 6], [7, 0, 3], [1, 3, 5], [6, 1, 2]]
    
    D = len(input[0])
    NSteps = 45

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
        ohm.Print()
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        output.Print()

        ohm.Step(data.lsbIn(), data.msbIn())
        
    output.PrintAll()
    
    return output.Output()


def test_OHM():
    ptf = "min"
    expectedmin = [-7, -7, -5, -6, -7, -7, -5]
    minr = runTest(ptf)    
    if minr != expectedmin:
        print(f"Expected: {expectedmin}")
        print(f"Got: {minr}")   
        assert False
    
    ptf = "max"
    expectedmax = [7, 7, 5, 6, 7, 7, 5]
    maxr = runTest(ptf)    
    if maxr != expectedmax:
        print(f"Expected: {expectedmax}")
        print(f"Got: {maxr}")   
        assert False

    ptf = "median"
    expectedmedian = [2, 0, 1, 1, 2, 0, 1]
    medr = runTest(ptf)
    if medr != expectedmedian:
        print(f"Expected: {expectedmedian}")
        print(f"Got: {medr}")   
        assert False

    print(f"Min: {minr}")
    print(f"Max: {maxr}")
    print(f"Median: {medr}")


test_OHM()
