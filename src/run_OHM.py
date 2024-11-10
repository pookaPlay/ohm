from bls.OHM_NET import RunNet


def test_RunNet():
    ptf = "median"
    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    expected = [2, 0, 1, 1, 2, 0, 1]    
    ret = RunNet(input, ptf)    
    
    if ret != expected:
        print(f"Expected: {expected}")
        print(f"Got: {ret}")   
        assert False    


test_RunNet()

