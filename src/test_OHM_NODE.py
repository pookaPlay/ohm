from bls.OHM_NODE_TEST_v4 import RunNode

def test_RunNode():
    ptf = "max"
    #ptf = "max"    
    NBitsIn = 4
    NBitsOut = 4
    NSteps = 16
    rep = 10
    
    
    input = [6, 1, 2]  # produces alternating 2 bit outputs
    input = [6, 7, 5]  # works with max    
    input = [input.copy() for _ in range(rep)]
    
    ret = RunNode(input, ptf, NBitsIn, NBitsOut, NSteps)    
    print(f"Ret: {ret}")


if __name__ == "__main__":
    
    test_RunNode()
    