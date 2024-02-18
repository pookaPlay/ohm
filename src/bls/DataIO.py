
def SerializeMSBOffset(input, NBits=8):
    # expects data to lie between -2^(NBits-1) and 2^(NBits-1)
    offset = 2**(NBits-1)
    input += offset

    thresholds = [2**i for i in range(NBits)]
    thresholds.reverse()
    
    output = list()
    for i in range(NBits):        
        #print(f"Step {i} : Input {input}")
        if input >= thresholds[i]:            
            output.append(1)
            input -= thresholds[i]
            #print("Bigger than " + str(thresholds[i]))
        else:
            output.append(0)

    return(output)            

def SerializeMSBTwos(input, NBits=8):
    # expects data to lie between -2^(NBits-1) and 2^(NBits-1)
    result = SerializeMSBOffset(input, NBits)
    # flip MSB
    result[0] = 1 - result[0]
    
    return(result)            

def SerializeMSBTwosSignExtend(input, NBits=8, extend=1):
    # expects data to lie between -2^(NBits-1) and 2^(NBits-1)
    result = SerializeMSBOffset(input, NBits)
    # flip MSB
    extendresult = [result[0]] * extend
    
    result = result + extendresult
        
    return(result)

def SerializeMSB(input, NBits=8):
    thresholds = [2**i for i in range(NBits)]
    thresholds.reverse()
    
    output = list()
    for i in range(NBits):        
        #print(f"Step {i} : Input {input}")
        if input >= thresholds[i]:            
            output.append(1)
            input -= thresholds[i]
            #print("Bigger than " + str(thresholds[i]))
        else:
            output.append(0)

    return(output)            

def DeserializeMSB(input):
    NBits= len(input)
    
    thresholds = [2**i for i in range(NBits)]
    thresholds.reverse()
    
    result = sum([input[i] * thresholds[i] for i in range(NBits)])

    return(result)

def DeserializeLSBTwos(input):
    NBits= len(input)
    offset = 2**(NBits-1)
    # flip MSB
    input[NBits-1] = 1 - input[NBits-1]
    thresholds = [2**i for i in range(NBits)]   
    #print("Decode")
    #print(input)
    #print(thresholds)
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset
    return(result)

def DeserializeMSBTwos(input):
    NBits= len(input)
    offset = 2**(NBits-1)
    # flip MSB
    input[0] = 1 - input[0]
    thresholds = [2**i for i in range(NBits)]   
    thresholds.reverse()
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset
    return(result)

def DeserializeMSBOffset(input):
    NBits= len(input)
    offset = 2**(NBits-1)

    thresholds = [2**i for i in range(NBits)]
    thresholds.reverse()
    
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset

    return(result)

def DeserializeLSBOffset(input):
    NBits= len(input)
    offset = 2**(NBits-1)

    thresholds = [2**i for i in range(NBits)]    
    
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset

    return(result)
