from OHM import OHM
from DataIO import DeserializeLSBTwos, DeserializeLSBOffset
from DataReader import DataReader
 
def test_OHM():

    NBitsIn = 4
    NBitsOut = 5
    input = [5, -2]
    D = len(input)
    NSteps = 15
    resultStart = 10
    resultEnd = 15

    data = DataReader(input, NBitsIn, NBitsOut)
    print(data.data)
    ohm = OHM(D, NBitsIn, NBitsOut)    
    result = list()   
    
    data.Reset()        
    ohm.Reset(data.Output())        

    print(f"== {0} ============================")

    ohm.Calc(data)
    result.append(ohm.Output())
    
    data.Print()
    ohm.Print("", 1)

    ohm.Step(data.isLsb(), data.isMsb())
        

    for bi in range(NSteps):
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        ohm.Calc(data)
        result.append(ohm.Output())
        ohm.Print("", 1)
        ohm.Step(data.isLsb(), data.isMsb())
    
    result = result[resultStart:resultEnd]
    print(result) 
    output = DeserializeLSBOffset(result)    
    print(output)
    
    return
        

test_OHM()