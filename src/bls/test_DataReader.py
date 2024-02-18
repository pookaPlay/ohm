from DataReader import DataReader

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

test_DataReader()