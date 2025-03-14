from DataIO import SerializeMSBOffset, DeserializeMSBOffset
import torch


#vector = torch.tensor([127, -127, 1, -1], dtype=torch.int32)
#num_bits = 8  # Number of bits for the two's complement representation
#twos_complement_tensor = vector_to_twos_complement(vector, num_bits)
#print(twos_complement_tensor)

STACK_BITS = 8

def PTF(input, weights, threshold):
    weighted_sum = sum(w for i, w in zip(input, weights) if i > 0)
    if weighted_sum >= threshold:     
        return 1    
    return 0    

weights = torch.ones(4, dtype=torch.float32)
threshold = 2
x = torch.tensor([[127, -64],[87, 99],[12, -45]], dtype=torch.float32)
print(x)

N = x.shape[0]
D = x.shape[1]
D2 = D*2
input_bits = torch.zeros(N, D2, STACK_BITS)
sticky_bits = torch.zeros(N, D2)
output_bits = torch.zeros(N, STACK_BITS)
output = torch.zeros(N)
#print(f'x: {x.shape}')

for ni in range(N):  
    for di in range(D):  
        val = SerializeMSBOffset(x[ni,di].item(), STACK_BITS)
        nval = [1-v for v in val]
        #print(val)
        input_bits[ni,di,:] = torch.tensor(val, dtype=torch.float32)
        input_bits[ni,D+di,:] = torch.tensor(nval, dtype=torch.float32)

    input_values = input_bits[ni,:,0]
    for k in range(STACK_BITS):        
        for di in range(D2):            
            if sticky_bits[ni,di] == 0:
                input_values[di] = input_bits[ni,di,k]
        
        output_bits[ni,k] = PTF(input_values, weights, threshold)
        
        for di in range(D2):            
            if sticky_bits[ni,di] == 0:
                if output_bits[ni,k] != input_bits[ni,di,k]:
                    sticky_bits[ni,di] = 1

    output[ni] = DeserializeMSBOffset(output_bits[ni,:].tolist())

print(output)