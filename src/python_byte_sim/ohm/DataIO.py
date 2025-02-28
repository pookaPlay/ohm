import torch

def int_to_twos_complement(value, num_bits):
    """Convert an integer to its two's complement binary representation."""
    if value < 0:
        value = (1 << num_bits) + value
    binary_str = format(value, f'0{num_bits}b')
    return [int(bit) for bit in binary_str]

def vector_to_twos_complement(vector, num_bits):
    """Convert a vector of integers to a tensor with the two's complement representation of each input."""
    twos_complement_list = [int_to_twos_complement(int(value), num_bits) for value in vector]
    return torch.tensor(twos_complement_list, dtype=torch.float32)

def twos_complement_to_int(twos_complement_vector):
    """Convert a two's complement vector back to an integer."""
    num_bits = twos_complement_vector.shape[-1]
    sign_bit = twos_complement_vector[..., 0]
    magnitude_bits = twos_complement_vector[..., 1:]
    
    # Convert magnitude bits to integer
    magnitude = magnitude_bits.matmul(2 ** torch.arange(num_bits - 1, dtype=torch.float32))
    
    # Apply sign
    result = magnitude - sign_bit * (2 ** (num_bits - 1))
    
    return result

def SerializeMSBOffset(data, NBits=8):
    
    if isinstance(data, int):
        input = data
    else:
        input = data.copy()
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

def SerializeLSBOffset(data, NBits=8):
    
    result = SerializeMSBOffset(data, NBits)
    result.reverse()
    return(result)

def SerializeMSBTwos(input, NBits=8):
    # expects data to lie between -2^(NBits-1) and 2^(NBits-1)
    result = SerializeMSBOffset(input, NBits)
    # flip MSB
    result[0] = 1 - result[0]
    
    return(result)            

def SerializeLSBTwos(input, NBits=8):
    # expects data to lie between -2^(NBits-1) and 2^(NBits-1)
    result = SerializeMSBOffset(input, NBits)
    # flip MSB
    result[0] = 1 - result[0]
    result.reverse()
    return(result)            

def SerializeMSBTwosSignExtend(input, NBits=8, extend=1):
    # expects data to lie between -2^(NBits-1) and 2^(NBits-1)
    result = SerializeMSBOffset(input, NBits)
    # flip MSB
    extendresult = [result[0]] * extend
    
    result = result + extendresult
        
    return(result)

def DeserializeMSBTwos(data):
    input = data.copy()
    NBits= len(input)
    offset = 2**(NBits-1)
    # flip MSB
    input[0] = 1 - input[0]
    thresholds = [2**i for i in range(NBits)]   
    thresholds.reverse()
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset
    return(result)

def DeserializeMSBOffset(data):
    input = data.copy()
    NBits= len(input)
    offset = 2**(NBits-1)

    thresholds = [2**i for i in range(NBits)]
    thresholds.reverse()
    
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset

    return(result)

def DeserializeLSBOffset(data):
    input = data.copy()
    NBits= len(input)
    offset = 2**(NBits-1)

    thresholds = [2**i for i in range(NBits)]    
    
    result = sum([input[i] * thresholds[i] for i in range(NBits)])
    result -= offset

    return(result)

def DeserializeLSBTwos(data):
    input = data.copy()
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
