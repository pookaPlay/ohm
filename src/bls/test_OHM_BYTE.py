from OHM_BYTE import OHM_BYTE


def test_lsb():

    numNodes = 1
    nodeDim = 2
    memK = 8
    memD = 8

    input = [3, 2, 1]
    weights = numNodes * [-5]    
    ptf = "max"
    #input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]
    #input = [[7, 7, 1, -6], [-2, 0, 3, 1], [-6, -3, 5, 2]]
    ohm = OHM_BYTE(memD, memK, numNodes, nodeDim, input, weights)
    
    ohm.RunNStep(1)
    result = ohm.GetLSBRead()
    print(f"LSB FINAL: {result}")    
    #if result != [6, 7, 8, 0, 0, 0, 0, 0]:
    #    print(f"LSB: {result}")
    #    assert(False)    

    return

test_lsb()

