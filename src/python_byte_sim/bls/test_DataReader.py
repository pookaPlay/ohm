from DataReader import DataReader
from DataIO import SerializeMSBTwos, DeserializeLSBTwos, DeserializeMSBTwos

def test_DataReader():
    # Test case 1
    input = [5, -7, 6]
    NBitsIn = 8
    NBitsOut = 9
    expected_output = [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0]]
    
    data_reader = DataReader(input, NBitsIn, NBitsOut)
    data_reader.Reset()
    data_reader.Print()
    for i in range(NBitsOut):
        data_reader.Step()
        data_reader.Print()
    #assert data_reader.Output() == expected_output


    print("All tests passed!")

def test_Serialize():
    input = -3
    testme = SerializeMSBTwos(input, 8)
    print(testme)
    result = DeserializeMSBTwos(testme)
    if input != result:
        print(f"Got {result} instead of {input}")

    input = 3
    testme = SerializeMSBTwos(input, 8)
    print(testme)
    result = DeserializeMSBTwos(testme)
    if input != result:
        print(f"Got {result} instead of {input}")


#test_DataReader()
test_Serialize()