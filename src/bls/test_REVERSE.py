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

test_REVERSE()