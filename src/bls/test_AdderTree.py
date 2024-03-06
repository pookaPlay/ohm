from OHM_BYTE import OHM_BYTE


def test_ohm_byte():

    numNodes = 4
    nodeDim = 2
    memK = 8
    memD = 8

    input = [3, 2, 1, 0]
    weights = numNodes * [1]    
    ptf = "max"
    #input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]
    #input = [[7, 7, 1, -6], [-2, 0, 3, 1], [-6, -3, 5, 2]]
    ohm = OHM_BYTE(memD, memK, numNodes, nodeDim, input, weights)
    
    ohm.RunNStep(1)
    ohm.PrintMem()
        
    return

test_ohm_byte()

