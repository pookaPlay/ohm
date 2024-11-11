from bls.OHM_NET_TEST import RunNet, RunNode


def test_RunNet():
    ptf = "median"
    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    expected = [2, 0, 1, 1, 2, 0, 1]    
    ret = RunNet(input, ptf)    
    
    if ret != expected:
        print(f"Expected: {expected}")
        print(f"Got: {ret}")   
        assert False    


def test_RunNode():
    ptf = "median"
    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    expected = [2, 0, 1, 1, 2, 0, 1]    
    ret = RunNode(input, ptf)    
    
    if ret != expected:
        print(f"Expected: {expected}")
        print(f"Got: {ret}")   
        assert False    

print(f"#######################################")
print(f"NODE TEST BEGIN")
print(f"#######################################")
test_RunNode()
print(f"#######################################")
print(f"NODE TEST END")
print(f"#######################################")

if 0:
    print(f"#######################################")
    print(f"NET TEST BEGIN")
    print(f"#######################################")
    test_RunNet()
    print(f"#######################################")
    print(f"NET TEST END")
    print(f"#######################################")

