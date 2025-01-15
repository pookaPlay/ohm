import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class OHMLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        
        lweight = weight.unsqueeze(0)  # (1, Cout, Cin*kH*kW, 1)
        x = lweight * input

        x = torch.sum(x, 2, keepdim=True)
        
        lbias = bias.unsqueeze(0)  # (1, Cout, Cin*kH*kW, 1)
        x = x + lbias

        return x         

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        lweight = weight.unsqueeze(0)
        grad_input = lweight * grad_output
        grad_input = grad_input.sum(1, keepdim=True)
        
        #grad_input = grad_input.unsqueeze(-1)

        grad_weight = grad_output * input

        grad_bias = grad_output

        return grad_input, grad_weight, grad_bias

class Linear(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, do_padding=False):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morpholigical neure. 
        kernel_size: scalar, the spatial size of the morpholigical neure.
        '''
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels*kernel_size*kernel_size), requires_grad=True)
        #nn.init.ones_(self.weight)
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
        #nn.init.constant_(self.bias, kernel_size*kernel_size*in_channels/2.0)
        #nn.init.constant_(self.bias, 1.0)
        nn.init.zeros_(self.bias)        
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
        self.padding = do_padding
        #self.outputThreshold = nn.Tanh()
        #self.outputThreshold = nn.ReLU()

    #def forward(self, x, weight=None, bias=None):
    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''        
        # padding
        if self.padding:
            x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches         
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))
        
        input = x.permute(1, 2, 0, 3)
        input = input.contiguous().view(input.shape[0], input.shape[1], -1)
        input = input.permute(2, 0, 1)
        input = input.contiguous()
        
        result = OHMLinearFunction.apply(input, self.weight, self.bias)
        result = result.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return result 

    def MyPrint(self):
        for p in self.parameters():
            print(p) 


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs
    
    
if __name__ == '__main__':
    # test
    print('Hellow world')


# class WOS(nn.Module):
#     def __init__(
#         self,
#         inputChannels=1,
#         outputChannels=1,
#         ksize=3,
#     ):
#         super(WOS, self).__init__()
#         self.ksize = ksize
#         block = []
#         block.append(nn.Conv2d(inputChannels, outputChannels, (ksize, ksize), stride=1, padding=(1,1)))
#         block.append(nn.Tanh())
#         #if batch_norm:
#         #    block.append(nn.BatchNorm2d(out_size))
#         self.block = nn.Sequential(*block)
        
#     def forward(self, X):
#         out = self.block(X)        
#         return out

