from PBF import PBF

def test_PBF():
    pbf = PBF(3)
    
    # Test case 1: All bits are 0
    x = [0, 0, 0]
    pbf.Step(x)
    assert pbf.Output() == 0

    # Test case 2: All bits are 1
    x = [1, 1, 1]
    pbf.Step(x)
    assert pbf.Output() == 1

    # Test case 3: Half of the bits are 1
    x = [1, 0, 1]
    pbf.Step(x)
    assert pbf.Output() == 1

    # Test case 4: More than half of the bits are 1
    x = [1, 1, 0]
    pbf.Step(x)
    assert pbf.Output() == 1

    # Test case 5: Less than half of the bits are 1
    x = [0, 1, 0]
    pbf.Step(x)
    assert pbf.Output() == 0

    # Test case 6: Random bits
    x = [1, 0, 1]
    pbf.Step(x)
    assert pbf.Output() == 1

    x = [0, 1, 0]
    pbf.Step(x)
    assert pbf.Output() == 0

    x = [1, 1, 1]
    pbf.Step(x)
    assert pbf.Output() == 1

    x = [0, 0, 0]
    pbf.Step(x)
    assert pbf.Output() == 0

    print("All tests passed!")

test_PBF()