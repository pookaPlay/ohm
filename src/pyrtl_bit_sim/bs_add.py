import pyrtl

def one_bit_add(a, b, carry_in):
    assert len(a) == len(b) == 1  # len returns the bitwidth
    sum = a ^ b ^ carry_in  # operators on WireVectors build the hardware
    carry_out = a & b | a & carry_in | b & carry_in
    return sum, carry_out

def bs_add(a, b):
    #carry_in = pyrtl.Register(bitwidth=1, name='carry')        
    carry_in = pyrtl.Register(bitwidth=1)        
    sum, carry_out = one_bit_add(a, b, carry_in)
    carry_in.next <<= carry_out
    return sum

# def bs_adder_tree(inputs):
#     D = len(inputs)            
#     numStart = D
#     tree = list()        
    
#     if numStart > 1:
#         tree.append([bs_add() for _ in range(numStart)])
#         numStart = int(numStart / 2)
#         while numStart > 1:
#             tree.append([bs_add() for _ in range(numStart)])
#             numStart = int(numStart / 2)            
                    
#         tree.append([bs_add()])   # top of binary tree 
        
#     tree.append([bs_add()])   # threshold       

