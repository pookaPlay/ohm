from RunWord import RunWord

def test_OHM_MSB():

    numNodes = 8
    nodeDim = 2
    memK = 8
    memD = 8

    input = [3, 2, 1, 0, -1, -2, -3, -4]
    weights = numNodes * [1]            
        
    ohm = RunWord(memD, memK, numNodes, nodeDim, input, weights)    
    ohm.Run()
    
    #ohm.PrintMem()
    #result = ohm.lsbMem.GetLSBInts()
    #print(result)
    
    
    return

test_OHM_MSB()