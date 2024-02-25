import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from FindMinWOS import FindIntWOS
import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# AQ7fPxRMAUxVpLSHbL_s
device = 'cpu'
ZERO_TOL = 1.0e-6

class ConvertToInteger(object):

    def __init__(self):
        pass

    def __call__(self, module):

        if hasattr(module, 'weight'):
            weight = module.weight.data
            bias = module.bias.data
            for fi in range(weight.shape[0]):
                w = weight[fi].detach().numpy()
                t = bias[fi].detach().numpy() 

                #print("WOS...")                
                #nw = w[w > ZERO_TOL]                                
                #print("Float WOS")
                #print(nw)
                #print(t)
                print("###################################################") 
                print("Real Valued: " + str(fi)) 
                print(w)
                print(t)
                WW, TT = FindIntWOS(w, t)
                print("----------------------------------------------------") 
                print("INT:") 
                print(WW)
                print(TT)

                #w[w > ZERO_TOL] = WW
                #t = TT
                weight[fi] = torch.tensor(WW)
                bias[fi] = torch.tensor(TT)
                #print("IWOS!!!")
                #print(w)
                #rint(t)
                
            #print(weight)
            #print(bias)
            module.weight.data = weight
            module.weight.bias = TT

class ClipNegatives(object):

    def __init__(self):
        pass

    def __call__(self, module):

        if hasattr(module, 'weight'):
            weight = module.weight.data
            bias = module.bias.data
            
            MD = weight.shape[1]        
            D = int(MD / 2)

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

            weight = torch.cat((newP, newN), 1)            
        
            nbias = -bias
            nbias[nbias < 0.0] = 0.0
            nbias = torch.true_divide(nbias, MD)

            weight = weight + nbias
            bias[bias < 0] = 0.0

            module.weight.data = weight
            module.weight.bias = bias


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
        NC = weight.shape[0]                
        D = weight.shape[1]

        lweight = weight.unsqueeze(0)
        lmask = mask.unsqueeze(0)

        mx = input + lmask
        
        allw = lweight.repeat(N, 1, 1)

        smx, si = torch.sort(mx, 2, descending=True)

        #print("Sorted Samples:")
        #print(smx)
        y = torch.zeros((N, NC))        
        lici = torch.zeros((N, NC), dtype=torch.int64)
        allnzw = torch.zeros((N, NC, D), dtype=torch.int64)
        #print("Selection")
        for ci in range(NC):
            ssi = si[:,ci,:].squeeze()
            saw = allw[:,ci,:].squeeze()
            if len(ssi.shape) == 1:
                ssi = ssi.unsqueeze(0)
                saw = saw.unsqueeze(0)

            sw = SmartSort(saw, ssi)            
            # need to take care of zero weight case  
            nzwIndex = sw > ZERO_TOL
            allnzw[:,ci,:] = nzwIndex

            rsw = sw[nzwIndex]
            numcols = int(rsw.shape[0] / N)            
            rsw = rsw.view( (N, numcols) ) 

            mysmx = smx[:,ci,:].squeeze()            
            if len(mysmx.shape) == 1:
                mysmx = mysmx.unsqueeze(0)
            rmysmx = mysmx[nzwIndex]
            rmysmx = rmysmx.view( (N, numcols) )             
            #print("Reduced mysmx:")
            #print(rmysmx.shape)

            accw = torch.cumsum(rsw, 1)
            #print("Samples:")
            #print(accw)
            #print(bias)
            li = torch.sum(torch.le(accw, bias[ci]), 1, keepdim=False)-1
            li[li < 0] = 0
            # need to adjust zero weight case back to original indicies
            #if nzw[wi]:
            #nzwIndex[upto] = wi
            #upto = upto + 1

            lici[:, ci] = li 
            #print("Selection:")
            #print(lici)

            yl = torch.gather(rmysmx, 1, lici[:,ci].unsqueeze(-1))
            y[:,ci] = yl.squeeze()
        
        ctx.lici = lici
        ctx.si = si
        ctx.allnzw = allnzw
        #print("output")
        #print(y)

        return y


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, mask, weight, bias = ctx.saved_tensors

        grad_input = grad_mask = grad_weight = grad_bias = None

        N = input.shape[0]
        NC = weight.shape[0]        
        MD = weight.shape[1]
                
        lici = ctx.lici
        si = ctx.si
        allnzw = ctx.allnzw

        grad_weight = torch.zeros([N, NC, MD])
        grad_bias = torch.zeros([N, NC, 1])
        grad_mask = torch.zeros([N, NC, MD])
        grad_input = torch.zeros([N, NC, MD])        
        
        for ci in range(NC):
            ssi = si[:,ci,:].squeeze()
            for i in range(0, N):               
                lii = lici[i, ci].item()
                liiz = 0                
                upto = -1
                for nzi in range(allnzw[i, ci].shape[0]):
                    if allnzw[i, ci, nzi] == True:
                        upto = upto + 1
                        if lii == upto:
                            liiz = nzi
                            break                
                #print("mapped " + str(lii) + " to " + str(liiz))
                #print(weight)
                lii = liiz
                

                elw = torch.cat((torch.arange(-(lii+1), 0), torch.arange(1, (MD-lii))))
                elw = torch.true_divide(elw, MD)
                abw = torch.cat((torch.arange(MD-lii, MD+1), torch.arange(MD-1, lii, -1)))
                abw = torch.true_divide(abw, MD)
                
                gradw  = -elw * grad_output[i, ci]                
                #print(ssi[i].shape)
                #print(gradw.shape)
                grad_weight[i, ci, ssi[i]] = gradw
                grad_bias[i, ci] = -elw.sum() * grad_output[i, ci]                

                #wi = ssi[i, lii]
                #grad_mask[i, ci, wi] = grad_output[i, ci]
                #grad_input[i, ci, wi] = grad_output[i, ci]
                
                gradm = abw * grad_output[i, ci]
                grad_mask[i, ci, ssi[i]] = gradm    

                grada = abw * grad_output[i, ci]
                grad_input[i, ci, ssi[i]] = grada
            
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
        
        #tempw = torch.Tensor(out_channels, D*2)
        #nn.init.uniform_(tempw, -1.0, 1.0)
        #tempw[:,D:(2*D)] = -tempw[:,0:D]
        #self.weight = nn.Parameter(tempw, requires_grad=True)

        self.weight = nn.Parameter(torch.Tensor(out_channels, D*2), requires_grad=True)
        #nn.init.ones_(self.weight)
        #nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.weight, 0.0, 1.0)

        self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
        temp = self.weight.sum(1)/2.0
        #print(temp.shape)
        #nn.init.uniform_(self.bias, -1.0, 1.0)
        
        nn.init.constant_(self.bias, temp.mean().item())
        #nn.init.constant_(self.bias, D+0.5)
        #nn.init.zeros_(self.bias)        

        self.mask = nn.Parameter(torch.Tensor(out_channels, D*2), requires_grad=True)
        #nn.init.zeros_(self.mask)        
        nn.init.uniform_(self.mask, -1.0, 1.0)

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
        
        #print(device)
        input = torch.cat((input, -input), 2)        
        #if len(input.shape) == 3:
        #    input = input.unsqueeze(-1)

        input = input.to(device)
        
        result = OHMWosFunction.apply(input, self.mask, self.weight, self.bias)        

        result = result.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return result

    #def backward(self, input):
    #    OHMWosFunction.verbose = self.verbose
    #    result = OHMWosFunction.apply(input, self.mask, self.weight, self.bias)

    def MyPrint(self):
        print("Weight: " + str(self.weight.data))
        print("Bias  : " + str(self.bias.data))
        print("Mask  : " + str(self.mask.data))

    def FindRanks(self, x, result):
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
        
        #print(device)
        input = torch.cat((input, -input), 2)        

        bits = 8
        dynRange = 2**(bits)
        print("Find ranks for " + str(bits) + " bits and range 0 -> " + str(dynRange-1))        
        NC = self.weight.shape[0] 
        D = self.weight.shape[1]
        
        lweight = self.weight.unsqueeze(0)
        lmask = self.mask.unsqueeze(0)
                
        mx = input + lmask        
        
        #print("input")
        #print(mx)
        #print("output")
        #print(result)

        dmx = mx.detach().numpy()
        minmax = torch.tensor([np.min(dmx), np.max(dmx)])
        nmx = (dynRange - 1.0) * (mx - minmax[0]) / (minmax[1] - minmax[0])
        clocks = result.detach().numpy()

        for ci in range(NC):
            #print(f"Chanel {ci} of {NC}") 
            w = lweight[:,ci,:].detach().numpy().squeeze()
            data = nmx[:,ci,:].squeeze()
            res = result[:,ci].squeeze()
            
            for n in range(data.shape[0]):                
                notOutput = np.zeros((D))
                notOutput[w < ZERO_TOL] = 1
                
                numNotOutput = sum(notOutput)
                currentBit = bits
                while (numNotOutput < D-1) and (currentBit > 0):
                    currentBit = currentBit - 1
                    ref = 2**(currentBit)
                    target = (res[n] > ref)
                    #print(target)
                    for di in range(D):
                        if notOutput[di] == False:
                            inval = data[n,di]  > ref
                            if inval != target:
                                notOutput[di] = True
                    numNotOutput = sum(notOutput)
                
                prec = bits - currentBit
                clocks[n, ci] = prec
        
        #print("Clock cycles: ")
        #print(clocks)
        return clocks


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
