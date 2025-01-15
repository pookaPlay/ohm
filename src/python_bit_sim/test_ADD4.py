from bls.ADD import ADD
from bls.DataIO import SerializeLSBOffset, DeserializeLSBOffset, SerializeLSBTwos, DeserializeLSBTwos

def test_ADD4():

    NBits = 4
    input = [-3, 7]
    print(input)
    #inputs = [SerializeLSBOffset(input[i], NBits) for i in range(len(input))]
    inputs = [SerializeLSBTwos(input[i], NBits) for i in range(len(input))]
        
    print(inputs)
        
    result = list()   
    adder = ADD()
    
    # first bit    
    bi = 0    
    adder.Calc(inputs[0][bi], inputs[1][bi], 1)
    output = adder.Output()
    result.append(output)
    inp = len(inputs[0])

    for bi in range(1, NBits+1):
        adder.Step()
        
        if bi < inp:
            adder.Calc(inputs[0][bi], inputs[1][bi], 0)
        else:
            adder.Calc(inputs[0][inp-1], inputs[1][inp-1], 0)
        
        output = adder.Output()
        result.append(output)        

    # Result is LSB
    print("################################")
    print(result)
    #output = DeserializeLSBOffset(result)        
    output = DeserializeLSBTwos(result)        
    print(output)


test_ADD4()
