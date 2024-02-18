from ADD import ADD

def test_ADD():    

    # Test case 1
    a = 1
    b = 1    
    
    adder = ADD()
    adder.Reset(a, b)
    adder.Step(a, b)
    
    #print(adder.Output())
    assert(adder.Output() == 0)
    adder.Print()
    

    

test_ADD()