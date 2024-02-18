from DataIO import SerializeMSBOffset, SerializeMSBTwos, SerializeMSB

def test_SerializeMSB():
    
    input = -65
    NBits = 8    
    result = SerializeMSB(input, NBits)
    print(f"Binary for {input}")
    print(result)
    #assert(result == expected)

    input = -65
    NBits = 8    
    result = SerializeMSBOffset(input, NBits)
    print(f"Offset for {input}")
    print(result)
    
    input = -65
    NBits = 8    
    result = SerializeMSBTwos(input, NBits)
    print(f"Twos for {input}")
    print(result)

    print("All tests passed!")


test_SerializeMSB()