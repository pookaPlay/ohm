import torch
import numpy as np

# Generate synthetic XOR data using four different Gaussians
def generate_xxor_data(n_samples):
    nper = int(n_samples/16)
    var = 0.05
    xt = -0.75
    yt = -0.75
    cls = 1
    allx = []
    ally = []
    for yi in range(4):
        cl = cls
        cls = cls * -1
        xt = -0.75
        for xi in range(4):
            xm = var * np.random.randn(nper, 2) + [xt, yt]
            
            if (yi < 2 and xi < 2) or (yi >= 2 and xi >= 2):
                cl = 1
            else:
                cl = -1

            ym = cl*np.ones(nper) 
            allx.append(xm)
            ally.append(ym)

            cl = cl * -1
            xt += 0.5

        yt += 0.5

    allx = np.vstack(allx)
    ally = np.hstack(ally)  
    return torch.tensor(allx, dtype=torch.float32), torch.tensor(ally, dtype=torch.float32)


# Generate synthetic XOR data using four different Gaussians
def generate_xor_data(n_samples):
    n_samples_per_quadrant = n_samples // 4
    ## was var = 16 offset 64 clip 128
    var = 0.1
    offset = 0.5
    clip = 1.
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, offset],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, -offset],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, offset],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, -offset]  # Bottom-right
    ])
    X = np.clip(X, -clip, clip)

    y = np.hstack([
        np.ones(n_samples_per_quadrant),  # Top-right
        np.ones(n_samples_per_quadrant),  # Bottom-left
        -np.ones(n_samples_per_quadrant),  # Top-left
        -np.ones(n_samples_per_quadrant)  # Bottom-right
    ])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Generate synthetic 3NOR data using four different Gaussians
def generate_3nor_data(n_samples, yind=0):
    n_samples_per_quadrant = n_samples // 4
    var = 0.1
    offset = 0.5
    clip = 1.
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, offset],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, -offset],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, offset],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, -offset]  # Bottom-right
    ])
    X = np.clip(X, -clip, clip)

    y = np.ones(n_samples_per_quadrant * 4)
    y[yind * n_samples_per_quadrant:(yind + 1) * n_samples_per_quadrant] = -1

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Generate synthetic linear data using four different Gaussians
def generate_linear_data(n_samples):
    n_samples_per_quadrant = n_samples // 4
    var = 0.1
    offset = 0.5
    clip = 1.
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, offset],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, -offset],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, offset],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, -offset]  # Bottom-right
    ])
    X = np.clip(X, -clip, clip)

    y = np.hstack([
        -np.ones(n_samples_per_quadrant),  # Top-right
        np.ones(n_samples_per_quadrant),  # Bottom-left
        np.ones(n_samples_per_quadrant),  # Top-left
        -np.ones(n_samples_per_quadrant)  # Bottom-right
    ])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


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
    input = data    
    #if isinstance(data, int):
    #    input = data
    #else:
    #    input = data.copy()
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
