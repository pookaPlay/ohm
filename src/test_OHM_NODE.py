from bls.OHM_NODE_TEST import RunNode


def test_RunNode():
    ptf = "median"
    input = [[101, 2, 35], [101, 2, 35], [101, 2, 35], [101, 2, 35]]
    expected = [2, 0, 1, 1, 2, 0, 1]    
    ret = RunNode(input, ptf)    
    print(f"Ret: {ret}")


print(f"#######################################")
print(f"NODE TEST BEGIN")
print(f"#######################################")
test_RunNode()
