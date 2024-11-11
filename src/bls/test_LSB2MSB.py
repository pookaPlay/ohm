from lsb2msb_v2 import lsb2msb_v2

def test_lsb2msb():
    test = lsb2msb_v2()
    a = [0, 0, 1, 0, 1, 0, 1, 1]
    b = [1, 1, 1, 0, 1, 0, 1, 1]
    
    print(f"Input {a}")    
    
    test.Reset()
    result = list()
    for i in range(len(a)):
        val = test.Output()
        result.append(val)
        
        test.Step(a[i])
        #print(f"Iter {i}: {test.Output()}")
    
    print(f"{result}")
    print(f"Switching")
    
    test.Switch()
    result2 = list()
    for i in range(len(a)):
        val = test.Output()
        result2.append(val)
        
        test.Step(b[i])
        #print(f"Iter {i}: {test.Output()}")

    print(f"{result2}")
    print(f"Switching")

    test.Switch()
    result3 = list()
    for i in range(len(a)):
        val = test.Output()
        result3.append(val)
        
        test.Step(a[i])
        #print(f"Iter {i}: {test.Output()}")
    
    print(f"{result3}")

test_lsb2msb()