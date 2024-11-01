import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def SmartSort(x, permutation):
    d1 = x.shape[0]
    d2 = x.shape[1]
    
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


class OHMWosFunction(Function):
    
    verbose = False

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, mask, weight, bias):        
        
        ctx.save_for_backward(input, mask, weight, bias)
        #ctx.mark_dirty(weight, bias)

        N = input.shape[0]
        #D = input.shape[2]
        NC = weight.shape[0]        
        MD = weight.shape[1]        
        D = int(MD / 2)
        # # adjust negatives
        if True:
            WPP = weight[:, 0:D]
            WPN = -WPP
            WPN[WPN < 0.0] = 0.0
            WPP[WPP < 0.0] = 0.0

            WNN = weight[:, D:2*D]
            WNP = -WNN
            WNP[WNP < 0.0] = 0.0
            WNN[WNN < 0.0] = 0.0
            
            newP = WPP + WNP
            newN = WPN + WNN
                    
        lweight = torch.cat((newP, newN), 1).unsqueeze(0)                        
        #lweight = weight.unsqueeze(0)        
        
        nbias = -bias
        nbias[nbias < 0.0] = 0.0
        bias[bias < 0.0] = 0.0
        nbias.unsqueeze(0)
        nbias = torch.true_divide(nbias, lweight.shape[2])
        
        lweight = lweight + nbias

        lmask = mask.unsqueeze(0)

        mx = input + lmask
        
        allw = lweight.repeat(N, 1, 1)

        if OHMWosFunction.verbose: 
            print("FWD: allw grad?")
            print(allw.requires_grad)

        smx, si = torch.sort(mx, 2, descending=True)
        y = torch.zeros((N, NC))        

        #print("Selection")
        for ci in range(NC):
            ssi = si[:,ci,:].squeeze()
            saw = allw[:,ci,:].squeeze()
            sw = SmartSort(saw, ssi)
            accw = torch.cumsum(sw, 1)
            
            li = torch.sum(torch.le(accw, bias[ci]), 1, keepdim=True)-1
            li[li < 0] = 0
            li[li >= MD] = MD-1
            #print(li)                        
            yl = torch.gather(smx[:,ci,:].squeeze(), 1, li)
            y[:,ci] = yl.squeeze()
        
        if OHMWosFunction.verbose: 
            print(y)

        return y


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, mask, weight, bias = ctx.saved_tensors

        grad_input = grad_mask = grad_weight = grad_bias = None

        N = input.shape[0]
        D = input.shape[2]        
        NC = weight.shape[0]        
        MD = weight.shape[1]

        lweight = weight.unsqueeze(0)
        lmask = mask.unsqueeze(0)
        
        mx = input + lmask                

        allw = lweight.repeat(N, 1, 1)
        
        smx, si = torch.sort(mx, 2, descending=True)

        diff = smx[:, :, 0:(MD-1)] - smx[:, :, 1:MD]
        diff = diff.cpu()
        sdiff = smx.sum(2, keepdim=True)
        diff0 = torch.zeros((smx.shape[0], smx.shape[1], 1))
        diffp = torch.cat( (diff0, diff),  2)
        diffn = torch.cat( (diff, diff0),  2)

        y = torch.zeros((N, NC))
        grad_weight = torch.zeros([N, NC, MD])
        grad_bias = torch.zeros([N, NC, 1])
        grad_mask = torch.zeros([N, NC, MD])
        grad_input = torch.zeros([N, NC, MD])
        
        
        #print("grad loop in " + str(NC) + " and " + str(N))
        #print(weight.shape)
        
        for ci in range(NC):
            #print(grad_output)
            #print("Prediction")
            ssi = si[:,ci,:].squeeze()
            saw = allw[:,ci,:].squeeze()
            ssmx = smx[:,ci,:].squeeze()
            csdiff = sdiff[:, ci].squeeze()
            cdiffn = diffn[:, ci, :].squeeze()            
            cdiffp = diffp[:, ci, :].squeeze()

            sw = SmartSort(saw, ssi)
            accw = torch.cumsum(sw, 1)

            li = torch.sum(torch.le(accw, bias[ci]), 1, keepdim=True)-1
            li[li < 0] = 0
            li[li >= MD] = MD-1
            #print(li)
            #test = li.squeeze()
            #minv = torch.min(test)
            #maxv = torch.max(test)                    
            #print("BWD Output Index: %f -> %f" % (minv, maxv))

            yl = torch.gather(ssmx, 1, li)
            y[:,ci] = yl.squeeze()
            
            for i in range(0, N):            
                lii = li[i].item()
                elw = torch.cat((torch.arange(-(lii+1), 0), torch.arange(1, (MD-lii))))
                elw = torch.true_divide(elw, MD)

                if True:
                    if grad_output[i, ci] < 0.0:
                        elw = elw * cdiffp[i]
                    else:
                        elw = elw * cdiffn[i]
                                
                gradw  = -elw * grad_output[i, ci]
                
                grad_weight[i, ci, ssi[i]] = gradw

                #grad_bias[i, ci] = -elw.sum() * grad_output[i, ci]
                grad_bias[i, ci] = -grad_output[i, ci]
                
                #abw = torch.cat((torch.arange(MD-lii, MD+1), torch.arange(MD-1, lii, -1)))
                #abw = torch.true_divide(abw, MD)                
                #grada = -abw * grad_output[i, ci]    
                #grad_mask[i, ci, ssi[i]] = grada
                #grad_input[i, ci, ssi[i]] = grada
                
                #gradb = welw.sum() * grad_output[i, ci]
                wi = ssi[i, lii]
                grad_mask[i, ci, wi] = grad_output[i, ci]
                grad_input[i, ci, wi] = grad_output[i, ci]
            
        print("BACK: grad input") 
        print(grad_input.shape)
        return grad_input.to(device), grad_mask.to(device), grad_weight.to(device), grad_bias.to(device)
        

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
        
        #tempw = torch.Tensor(out_channels, D*2)
        #nn.init.uniform_(tempw, -1.0, 1.0)
        #tempw[:,D:(2*D)] = -tempw[:,0:D]
        #self.weight = nn.Parameter(tempw, requires_grad=True)

        self.weight = nn.Parameter(torch.Tensor(out_channels, D*2), requires_grad=True)
        nn.init.ones_(self.weight)
        #nn.init.xavier_uniform_(self.weight)
        #nn.init.uniform_(self.weight, -1.0, 1.0)

        self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
        #temp = self.weight.sum(1)/2.0
        #print(temp.shape)
        #nn.init.uniform_(self.bias, -1.0, 1.0)
        #nn.init.constant_(self.bias, temp.item())
        nn.init.constant_(self.bias, D+0.5)
        #nn.init.zeros_(self.bias)        

        self.mask = nn.Parameter(torch.Tensor(out_channels, D*2), requires_grad=True)
        nn.init.zeros_(self.mask)        
        #nn.init.uniform_(self.mask, -1.0, 1.0)

        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
        self.padding = do_padding
        self.verbose = False


    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''                
        # padding  
        OHMWosFunction.verbose = self.verbose         

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
        
        #print("MOD FWD")
        #print(x.device.index)
        #print(device)
        input = torch.cat((input, -input), 2)                
        input = input.to(device)
        
        result = OHMWosFunction.apply(input, self.mask, self.weight, self.bias)
        #result, tweight, tbias = OHMWosFunction.apply(input, self.mask, self.weight, self.bias)
        #self.weight = tweight.clone()
        #self.bias = tbias.clone()
        #if self.verbose:
        #    print(result)
        result = result.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return result 

    #def backward(self, input):
    #    OHMWosFunction.verbose = self.verbose
    #    result = OHMWosFunction.apply(input, self.mask, self.weight, self.bias)

    def MyPrint(self):
        print("Weight: " + str(self.weight.data))
        print("Bias  : " + str(self.bias.data))
        print("Mask  : " + str(self.mask.data))
            


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
