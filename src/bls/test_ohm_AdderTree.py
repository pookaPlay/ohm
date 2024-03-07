from RunByte import RunByte

def test_OHM_AdderTree_bias():

    numNodes = 4
    # This should be 1 or numNodes
    numNodeOutputs = 4
    
    nodeDim = 2
    memK = 8
    memD = 8

    input = [3, 2, 1, 0]
    weights = numNodes * [-1]            
    
    expected = [2, 1, 0, -1]

    ohm = RunByte(memD, memK, numNodes, numNodeOutputs, nodeDim, input, weights)
    
    ohm.RunNStep(1)
    #ohm.PrintMem()
    result = ohm.lsbMem.GetLSBInts()
    #print(result)
    
    if (result[0:numNodeOutputs] != expected):
        print(f"Failed test: got {result} expected {expected}")
        assert False
    
    return

def test_OHM_AdderTree_ptf():

    numNodes = 4
    # This should be 1 or numNodes
    numNodeOutputs = 1
    
    nodeDim = 2
    memK = 8
    memD = 8

    input = [3, 2, 1, 0]
    weights = numNodes * [1]            
    numNodeOutputs = 1    
    expected = 10

    ohm = RunByte(memD, memK, numNodes, numNodeOutputs, nodeDim, input, weights)
    
    ohm.RunNStep(1)
    #ohm.PrintMem()
    result = ohm.lsbMem.GetLSBInts()
    if (result[0] != expected):
        print(f"Failed test: got {result[0]} expected {expected}")
        assert False
    

    return
