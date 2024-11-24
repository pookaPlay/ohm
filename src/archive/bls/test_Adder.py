from CCAdder import CCAdder
from DataIO import SerializeMSBTwos, DeserializeMSBTwos, DeserializeLSBTwos, SerializeMSBTwosSignExtend
import pytest

""" @pytest.fixture
def sample_data():
    NBits = 8
    SignExtend = 1
    input = [5, 4]
    inputs = [SerializeMSBTwosSignExtend(input[i], NBits, SignExtend) for i in range(len(input))]
    return {"NBits": NBits, "inputs": inputs, "expected": 9}
 """
def test_Adder():

    NBits = 8
    input = [5, 4]
    expected = 10
    #inputs = [SerializeMSBOffset(input[i], NBits) for i in range(len(input))]
    inputs = [SerializeMSBTwosSignExtend(input[i], NBits, 1) for i in range(len(input))]
    print(input)
    print(inputs)
        
    result = list()   
    add = CCAdder(NBits)
    add.Reset(inputs[0], inputs[1])
    
    # first bit    
    bi = 0
    output = add.Output()
    result.append(output)
    print(f"{bi+1}: {result}")
    
    for bi in range(NBits-1):
        add.Step()
        output = add.Output()
        result.append(output)
        print(f"{bi+1}: {result}")
        add.Print()

    # Result is LSB
    print("################################")
    print(result)
    output = DeserializeLSBTwos(result)    
    assert output == expected

    print(output)


test_Adder()
