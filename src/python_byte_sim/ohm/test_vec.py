from DataIO import vector_to_twos_complement
import torch


vector = torch.tensor([127, -127, 1, -1], dtype=torch.int32)
num_bits = 8  # Number of bits for the two's complement representation
twos_complement_tensor = vector_to_twos_complement(vector, num_bits)
print(twos_complement_tensor)