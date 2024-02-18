from REVERSE import REVERSE
from DataIO import SerializeMSB

def test_REVERSE():

    NBits = 8            
    input = SerializeMSB(13, NBits)    
    print(input)    

    result = list()
    
    reverse = REVERSE(NBits)
    reverse.Reset()    
    # first bit
    result.append(reverse.Output())

    for bi in range(NBits*3):
        if bi < NBits:
            reverse.Step(input[bi])
        else:
            reverse.Step(0)
        
        result.append(reverse.Output())
        print(result)

    # Result is LSB
    print(result)
    
    return



    NBits = 8

    # Test case 1
    reverse = REVERSE(NBits)
    reverse.Reset()
    reverse.Step(1)
    reverse.Step(0)
    reverse.Step(1)
    reverse.Step(0)
    reverse.Step(1)
    reverse.Step(0)
    reverse.Step(1)
    reverse.Step(0)
    output = reverse.Output()
    assert output == 0, f"Test case 1 failed. Expected output: 0, Actual output: {output}"

    # Test case 2
    reverse.Reset()
    reverse.Step(0)
    reverse.Step(1)
    reverse.Step(0)
    reverse.Step(1)
    reverse.Step(0)
    reverse.Step(1)
    reverse.Step(0)
    reverse.Step(1)
    output = reverse.Output()
    assert output == 1, f"Test case 2 failed. Expected output: 1, Actual output: {output}"

    # Test case 3
    reverse.Reset()
    reverse.Step(1)
    reverse.Step(1)
    reverse.Step(1)
    reverse.Step(1)
    reverse.Step(1)
    reverse.Step(1)
    reverse.Step(1)
    reverse.Step(1)
    output = reverse.Output()
    assert output == 1, f"Test case 3 failed. Expected output: 1, Actual output: {output}"

    # Test case 4
    reverse.Reset()
    reverse.Step(0)
    reverse.Step(0)
    reverse.Step(0)
    reverse.Step(0)
    reverse.Step(0)
    reverse.Step(0)
    reverse.Step(0)
    reverse.Step(0)
    output = reverse.Output()
    assert output == 0, f"Test case 4 failed. Expected output: 0, Actual output: {output}"

    print("All test cases passed!")

test_REVERSE()