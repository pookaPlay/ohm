import torch
import numpy as np

# DL_FUNCTIONS = [
#     "zero",
#     "and",
#     "not_implies",
#     "a",
#     "not_implied_by",
#     "b",
#     "xor",
#     "or",
#     "not_or",
#     "not_xor",
#     "not_b",
#     "implied_by",
#     "not_a",
#     "implies",
#     "not_and",
#     "one",
# ]
# def GetFunctionText(D, i):   
#     return DL_FUNCTIONS[i]

# def GetNumFunctions(D):
#     return 16

# def multi_op(inputs, i):
#     D = len(inputs)
#     assert D == 2
#     a = inputs[0]
#     b = inputs[1]

#     if i == 0:
#         return torch.zeros_like(a)
#     elif i == 1:
#         return a * b
#     elif i == 2:
#         return a - a * b
#     elif i == 3:
#         return a
#     elif i == 4:
#         return b - a * b
#     elif i == 5:
#         return b
#     elif i == 6:
#         return a + b - 2 * a * b
#     elif i == 7:
#         return a + b - a * b
#     elif i == 8:
#         return 1 - (a + b - a * b)
#     elif i == 9:
#         return 1 - (a + b - 2 * a * b)
#     elif i == 10:
#         return 1 - b
#     elif i == 11:
#         return 1 - b + a * b
#     elif i == 12:
#         return 1 - a
#     elif i == 13:
#         return 1 - a + a * b
#     elif i == 14:
#         return 1 - a * b
#     elif i == 15:
#         return torch.ones_like(a)


# def multi_op_s(inputs, i_s):        
#     numFuncs = GetNumFunctions(len(inputs))

#     r = torch.zeros_like(inputs[0])
#     for i in range(numFuncs):
#         u = multi_op(inputs, i)
#         r = r + i_s[..., i] * u

#     return r

def GetFunctionText(D, i):   
    D2 = 2*D
    if i < D:
        ret =f'In{i}'
    elif i < D2:
        ret = f'-In{i-D}'
    elif i == D2:
        ret = 'zero'
    elif i == D2+1:
        ret = 'one'
    elif i == D2+2:
        ret = 'prod'
    elif i == D2+3:
        ret = 'sum-prod'
    elif i == D2+4:
        ret = '1-prod'
    elif i == D2+5:
        ret = '1-(sum-prod)'
    return ret

def GetNumFunctions(D):
    return 2*D+6

def multi_op(inputs, index):    
    D = len(inputs)
    D2 = 2*D
    inputs_tensor = torch.stack(inputs, dim=0)
    #print(f'inputs_tensor: {inputs_tensor.shape}')
    prodval = torch.prod(inputs_tensor, dim=0)
    sumval = torch.sum(inputs_tensor, dim=0)
    maxval = torch.max(inputs_tensor, dim=0).values
    minval = torch.min(inputs_tensor, dim=0).values
    #print(f'From {D}, prodval: {prodval}, sumval: {sumval}, maxval: {maxval}, minval: {minval}')

    if index < D:
        return inputs[index]
    elif index < D2:
        return 1 - inputs[index-D]
    elif index == D2: 
        return torch.zeros_like(inputs[0])
    elif index == D2+1:
        return torch.ones_like(inputs[0])    
    elif index == D2+2:        
        return prodval
    elif index == D2+3:        
        return sumval-prodval
    elif index == D2+4:        
        return 1-prodval
    elif index == D2+5:        
        return 1-(sumval-prodval)
    
    

def multi_op_s(inputs, i_s):        
    numFuncs = GetNumFunctions(len(inputs))
    
    r = torch.zeros_like(inputs[0])
    for i in range(numFuncs):
        u = multi_op(inputs, i)
        r = r + i_s[..., i] * u
    
    return r


#######################################################################################################################
# DL_FUNCTIONS = [
#     "zero",
#     "and",
#     "not_implies",
#     "a",
#     "not_implied_by",
#     "b",
#     "xor",
#     "or",
#     "not_or",
#     "not_xor",
#     "not_b",
#     "implied_by",
#     "not_a",
#     "implies",
#     "not_and",
#     "one",
# ]


# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |  <=
# | 1  | A and B              | 0     | 0     | 0     | 1     |  <=
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |  <=
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |  <=
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |  <=
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |  <=

# def bin_op(a, b, i):
#     assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
#     if a.shape[0] > 1:
#         assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

#     if i == 0:
#         return torch.zeros_like(a)
#     elif i == 1:
#         return a * b
#     elif i == 2:
#         return a - a * b
#     elif i == 3:
#         return a
#     elif i == 4:
#         return b - a * b
#     elif i == 5:
#         return b
#     elif i == 6:
#         return a + b - 2 * a * b
#     elif i == 7:
#         return a + b - a * b
#     elif i == 8:
#         return 1 - (a + b - a * b)
#     elif i == 9:
#         return 1 - (a + b - 2 * a * b)
#     elif i == 10:
#         return 1 - b
#     elif i == 11:
#         return 1 - b + a * b
#     elif i == 12:
#         return 1 - a
#     elif i == 13:
#         return 1 - a + a * b
#     elif i == 14:
#         return 1 - a * b
#     elif i == 15:
#         return torch.ones_like(a)


########################################################################################################################


def get_unique_connections(in_dim, out_dim, device='cuda'):
    assert out_dim * 2 >= in_dim, 'The number of neurons ({}) must not be smaller than half of the number of inputs ' \
                                  '({}) because otherwise not all inputs could be used or considered.'.format(
        out_dim, in_dim
    )

    x = torch.arange(in_dim).long().unsqueeze(0)

    # Take pairs (0, 1), (2, 3), (4, 5), ...
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]

    # If this was not enough, take pairs (1, 2), (3, 4), (5, 6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # If this was not enough, take pairs with offsets >= 2:
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])

    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)

    perm = torch.randperm(out_dim)

    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)

    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    a, b = a.contiguous(), b.contiguous()
    return a, b


########################################################################################################################


class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None


########################################################################################################################


BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}

