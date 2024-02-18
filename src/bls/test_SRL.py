from SRL import SRL

def test_SRL():
    srl = SRL(8, 1)
    a = [0, 0, 0, 0, 0, 0, 1, 1]        
    srl.Reset(a)
    result = list()
    for i in range(8):        
        result.append(srl.Output())
        #print(f"Iter {i}: {result}")
        srl.Step()        
        
    print(result)
    

test_SRL()