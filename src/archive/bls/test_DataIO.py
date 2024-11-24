from DataIO import SerializeMSBOffset, SerializeMSBTwos

def test_SerializeMSB():
    
    input = -3
    NBits = 8    
    result = SerializeMSBTwos(input, NBits)
    print(f"Binary for {input}")
    print(result)
    #assert(result == expected)
    
    input = 3
    NBits = 8    
    result = SerializeMSBTwos(input, NBits)
    print(f"Twos for {input}")
    print(result)

    print("All tests passed!")


test_SerializeMSB()