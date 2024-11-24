from bls.ADD import ADD
from bls.DataIO import SerializeLSBOffset, DeserializeLSBOffset, SerializeLSBTwos, DeserializeLSBTwos

def test_ADD4():

    NBits = 4
    input = [2, -3]
    print(input)
    inputs = [SerializeLSBOffset(input[i], NBits) for i in range(len(input))]
    #inputs = [SerializeLSBTwos(input[i], NBits) for i in range(len(input))]
        
    print(inputs)
        
    result = list()   
    adder = ADD()
    
    # first bit    
    bi = 0    
    adder.Calc(inputs[0][bi], inputs[1][bi], 1)
    output = adder.Output()
    result.append(output)
    
    for bi in range(NBits-1):
        adder.Step()

        adder.Calc(inputs[0][bi+1], inputs[1][bi+1], 0)
        output = adder.Output()
        result.append(output)        

    # Result is LSB
    print("################################")
    print(result)
    #output = DeserializeLSBOffset(result)        
    output = DeserializeLSBTwos(result)        
    print(output)


test_ADD4()
