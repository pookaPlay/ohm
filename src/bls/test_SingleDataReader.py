from SingleDataReader import SingleDataReader

def test_SingleDataReader():
    
    input = [[7, 7, 1, -6], [-2, 0, 3, 1], [-6, -3, 5, 2]]
    Kin = 7
    Kout = 8
    D = len(input)   
    ni = 0
    dataMem = [SingleDataReader(input[ni], Kin, Kout) for ni in range(D)]
    nsteps = 100

    [dataMem[_].Reset() for _ in range(D)]
    [dataMem[_].Print() for _ in range(D)]

    for i in range(nsteps):
        [dataMem[_].Step() for _ in range(D)]
        [dataMem[_].Print() for _ in range(D)]    

    

def test_SingleDataReaderOne():
    
    input = [7, 7, 1, -6]
    Kin = 7
    Kout = 8
    
    ni = 0
    dataMem = SingleDataReader(input, Kin, Kout)
    nsteps = 100

    dataMem.Reset()
    dataMem.Print()

    for i in range(nsteps):
        dataMem.Step()
        dataMem.Print()

    

test_SingleDataReaderOne()
test_SingleDataReader()