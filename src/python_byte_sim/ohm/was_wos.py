import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class WOS(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morpholigical neure. 
        kernel_size: scalar, the spatial size of the morpholigical neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(WOS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        #nn.init.ones_(self.weight)
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
        #nn.init.constant_(self.bias, kernel_size*kernel_size*in_channels/2.0)        
        #nn.init.constant_(self.bias, -1.0)        
        nn.init.zeros_(self.bias)        
        self.mask = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        nn.init.uniform_(self.mask, -1.0, 1.0)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
        self.threshold = nn.Tanh()
        #self.outputThreshold = nn.Tanh()
        #self.outputThreshold = nn.ReLU()

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        ctx.save_for_backward(x, self.mask, self.weight)
        # padding
        #x = fixed_padding(x, self.kernel_size, dilation=1)
        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # apply additive weights
        mask = self.mask.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        # threshold
        x = self.threshold(mask + x) # (B, Cout, Cin*kH*kW, L)

        # apply multiplicative weights
        weight = self.weight.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)
        x = weight * x

        # sum
        x = torch.sum(x, 2)
        
        # apply bias 
        x = x - self.bias
        # and threshold        
        #minv = torch.min(x)
        #maxv = torch.max(x)        
        #print(" FWD: Prior to output %f -> %f" % (minv, maxv))

        #x = self.outputThreshold(x) 

        #minv = torch.min(x)
        #maxv = torch.max(x)        
        #print(" FWD: After tanh %f -> %f" % (minv, maxv))

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)
        return x 
    
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
