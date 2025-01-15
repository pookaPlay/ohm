import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function



def SmartSort(x, permutation):
    d1 = x.shape[0]
    d2 = x.shape[1]
    
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


class OHMWosFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, mask, weight, bias):        
        
        ctx.save_for_backward(input, mask, weight, bias)

        print("FWD: input")
        print(input.shape)
        #print("D: " + str(D))
        #print("###########################")
        #print("###########################")
        #print("FWD: Weight")
        #print(weight)
        #print(bias)         
        #print("FWD: Mask")
        #print(mask)

        mw = weight.unsqueeze(0).unsqueeze(-1)
        mask = mask.unsqueeze(0).unsqueeze(-1)
        
        print("Mask:") 
        print(mask.shape)
        #print("FWD: input")
        #print(input.shape)
        #print(input[0])
        x = mask + input
        
        mx = torch.cat((x, -x), 2)
        D = mx.shape[2]
        #print(mx[0])                
        #print(mx.shape)
        #mw = torch.cat((weight, weight), 2)    
        #print("FWD: weight")
        #print(mw.shape)
        allw = mw.repeat(input.shape[0], 1, 1, 1)

        smx, si = torch.sort(mx, 2, descending=True)
        
        smx = smx.squeeze()
        allw = allw.squeeze()
        si = si.squeeze()
        print("Smart sort option")
        print(si.shape)
        print(allw.shape)

        sw = SmartSort(allw, si)

        accw = torch.cumsum(sw, 1)
        #print("Cummulative")
        #print(accw)

        li = torch.sum(torch.le(accw, bias), 1, keepdim=True)-1
        li[li < 0] = 0
        li[li >= D] = D-1
        #print(li)
        #exit()
        y = torch.gather(smx, 1, li)
        
        #print("---------------------------")
        #print("FWD Result:")
        #print(y.squeeze())
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, mask, weight, bias = ctx.saved_tensors
        
        grad_input = grad_mask = grad_weight = grad_bias = None

        #print("On backward grad_output: ")        
        #print(grad_output)
        #print(input.shape)
        #print(weight.shape)
        N = input.shape[0]
        D = weight.shape[1]
        M = mask.shape[1]
        grad_weight = torch.zeros([N, D])
        grad_bias = torch.zeros([N, 1])
        grad_mask = torch.zeros([N, M])
        
        mw = weight.unsqueeze(0).unsqueeze(-1)
        mask = mask.unsqueeze(0).unsqueeze(-1)
        x = mask + input

        mx = torch.cat((x, -x), 2)
        allw = mw.repeat(input.shape[0], 1, 1, 1)

        smx, si = torch.sort(mx, 2, descending=True)
        
        smx = smx.squeeze()
        allw = allw.squeeze()
        si = si.squeeze()
        
        sw = SmartSort(allw, si)

        accw = torch.cumsum(sw, 1)
        li = torch.sum(torch.le(accw, bias), 1, keepdim=True)-1
        li[li < 0] = 0
        li[li >= D] = D-1

        y = torch.gather(smx, 1, li)   #.type(torch.Long)

        #print("BACKWARD =========>")
        #print(si)                                        
        for i in range(0, N):            
            elw = torch.cat((torch.arange(-(li[i].item()+1), 0), torch.arange(1, (D-li[i].item()))))            
            elw = elw * grad_output[i]
            grad_weight[i, si[i]] = -elw
            # Get the input index             
            wi = si[i, li[i].item()]            
            ws = 1
            if wi >= M:
                wi = D - wi - 1
                ws = -1
            grad_mask[i, wi] = grad_output[i] * ws

        
        #print(grad_weight)
        #print(shiftr)
        #print(grad_weight)
        #grad_bias = 0.0        
        #diffs = mx[:,:,1:D,:] - mx[:,:,0:D-1,:]

        #print(accw)
        #print(accw)
        #deltas = torch.abs(bias - accw)
        ##print(deltas)
        #maxDelta = torch.max(deltas)
        #print("maxDelta")
        #print(maxDelta)        
        #deltas = torch.div(deltas, maxDelta)
        #deltas = 1 - deltas
        #print(deltas)        
        
        return grad_input, grad_mask, grad_weight, grad_bias

class WOS(nn.Module):
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
        super(WOS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        D = in_channels*kernel_size*kernel_size
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, D*2), requires_grad=True)
        nn.init.ones_(self.weight)
        #nn.init.uniform_(self.weight, 1.0, 2*D)        
        
        self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
        #nn.init.constant_(self.bias, kernel_size*kernel_size*in_channels/2.0)
        nn.init.constant_(self.bias, D+0.5)
        #nn.init.zeros_(self.bias)        

        self.mask = nn.Parameter(torch.Tensor(out_channels, in_channels*kernel_size*kernel_size), requires_grad=True)
        nn.init.zeros_(self.mask)        
        #nn.init.uniform_(self.mask, -1.0, 1.0)

        #self.sign = torch.cat((torch.ones((1, D)), -torch.ones((1,D))), 1)
        #self.threshold = D

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
        #print("Input")
        #print(x.shape)
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches         
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        #print("After unfold")
        #print(x.shape)

        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        result = OHMWosFunction.apply(x, self.mask, self.weight, self.bias)
        result = result.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)
        #print("Final output")
        #print(result.shape)

        return result 

    def MyPrint(self):
        print("Weight: " + str(self.weight))
        print("Bias  : " + str(self.bias))
        print("Mask  : " + str(self.mask))
            

    def Update(self, error, input):
        print("Update")
        #print(input.shape)
        #input input.squeeze()
        #self.weight = self.weight.add(self.sign * error)
        #self.weight[ self.weight < 0.0 ] = 0.0
        #print(self.weight)


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
