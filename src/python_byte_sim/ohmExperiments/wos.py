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
WOS_PRECISION = 16

def GetRefs(res, bits):
    refs = list()
    dirs = list()
    mybit = bits    
    tref = 0.0 
    
    while mybit > 0:
        mybit = mybit - 1
        ref = 2**mybit
        tref = tref + ref
        refs.append(tref)        
        if res < tref:
            tref = tref - ref
            dirs.append(False)
        else:
            dirs.append(True)
    
    return refs, dirs

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
    gradWs = dict()
    gradMs = dict()
    gradIs = dict()
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, mask, weight, bias):        
        
        ctx.save_for_backward(input, mask, weight, bias)

        #print("On FWD")
        #print(weight)
        #print(bias)
        N = input.shape[0]        
        NC = weight.shape[0]                
        D = weight.shape[1]

        lmask = mask.unsqueeze(0)
        lweight = weight.unsqueeze(0)
        
        y = torch.zeros((N, NC))        
        lici = torch.zeros((N, NC), dtype=torch.int64)
        si = torch.zeros((N, NC, D), dtype=torch.int64)
        numZero = torch.zeros((NC), dtype=torch.int64)

        for ci in range(NC):
            #print("Channel " + str(ci))    
            mx = input + lmask[0,ci]
            myw =  lweight[0,ci]
            nzw = myw > ZERO_TOL            
            
            nzwi = torch.tensor([i for i, val in enumerate(nzw) if val])
            if (nzwi.shape[0] == 0):
                print("Unexpected Case when all weights are zero")
                exit()            
            nzwi = nzwi.unsqueeze(0)
            allnzwi = nzwi.repeat(N, 1)

            zwi = torch.tensor([i for i, val in enumerate(nzw) if not val])            
            numZero[ci] = zwi.shape[0]
            if numZero[ci] > 0:
                zwi = zwi.unsqueeze(0)            
                allzwi = zwi.repeat(N,1)
            
            rw = myw[nzw]
            allw = rw.repeat(N, 1)
            nzmx = mx[:,0, nzw]                        
            
            smx, ssi = torch.sort(nzmx, 1, descending=True)
            #print("allnzwi")
            #print(allnzwi)
            #print("ssi")
            #print(ssi)
            #print("smx")
            #print(smx)

            nzssi = SmartSort(allnzwi, ssi)
            #print("nzssi")
            #print(nzssi)
            if numZero[ci] > 0:
                allssi = torch.cat((nzssi, allzwi), 1)
            else:
                allssi = nzssi

            #print("allssi")
            si[:,ci,:] = allssi

            sw = SmartSort(allw, ssi)
            accw = torch.cumsum(sw, 1)
            li = torch.sum(torch.le(accw, bias[ci]), 1, keepdim=False)-1
            li[li < 0] = 0            
            lici[:, ci] = li 
            
            yl = torch.gather(smx, 1, li.unsqueeze(-1)) 
            y[:,ci] = yl.squeeze()                       
            # This adjusts from sorted to original indicies
            #lii = torch.gather(nzssi, 1, li).squeeze()         
            #lici[:, ci] = lii            
            #nlii = lii.unsqueeze(-1).unsqueeze(-1)            
            #ytest = torch.gather(mx, 2, nlii)
            
        #print("Output")
        #print(lici.squeeze())
        #print(y.squeeze())
        #print("si")
        #print(si)
        #print("input")
        #print(mx)

        ctx.lici = lici  # Index of output in original space
        ctx.si = si      # Indicies of complete sorted input
        ctx.numZero = numZero
        return y


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, mask, weight, bias = ctx.saved_tensors        
        grad_input = grad_mask = grad_weight = grad_bias = None

        #print("On Backward")        
        #print(grad_output.squeeze())
        
        #print(weight)
        #print(bias)

        N = input.shape[0]
        NC = weight.shape[0]        
        MD = weight.shape[1]
        #print("In backward with MD " + str(MD) + " and NC " + str(NC))

        lici = ctx.lici
        si = ctx.si
        numZero = ctx.numZero
        #print("Indexing")
        #print(lici.shape)
        #print(lici)
        #print(si.shape)
        #elici = (si == lici).nonzero()
        #li = elici[:,2]
        #print(li)

        grad_weight = torch.zeros([N, NC, MD])
        grad_bias = torch.zeros([N, NC, 1])
        grad_mask = torch.zeros([N, NC, MD])
        grad_input = torch.zeros([N, NC, MD])        
        tweight = torch.zeros([N, NC, MD])        
        
        tmask = torch.zeros([N, NC, MD])
        tinput = torch.zeros([N, NC, MD])        
        #tmask = torch.zeros([N, NC, 1])
        #tinput = torch.zeros([N, NC, 1])        
        #tsel = torch.zeros([N, NC, 1], dtype=torch.int64)        

        allws = OHMWosFunction.gradWs[MD].repeat(N, 1, 1)
        allms = OHMWosFunction.gradMs[MD].repeat(N, 1, 1)
        allis = OHMWosFunction.gradIs[MD].repeat(N, 1, 1)
        #print("Grad patterns")
        #print(OHMWosFunction.gradWs.shape)
        #print(allws.shape)
        
        for ci in range(NC):
            ssi = si[:,ci,:].squeeze()
            lii = lici[:, ci].unsqueeze(-1).unsqueeze(-1)
            li = lii.repeat(1, 1, MD)
            myws = torch.gather(allws, 1, li)            
            myms = torch.gather(allms, 1, li)
            myis = torch.gather(allis, 1, li)
            
            #myws[:, 0, torch.arange(MD-numZero[ci],MD)] = 0.0
            #myms[:, 0, torch.arange(MD-numZero[ci],MD)] = 0.0
            #myis[:, 0, torch.arange(MD-numZero[ci],MD)] = 0.0
            
            gradw = myws[:,0,:] * grad_output[:, ci].unsqueeze(-1)                                    
            tweight[:,ci,:] = -gradw            

            mysum = myws.sum(2)            
            gradb =  mysum * grad_output[:, ci].unsqueeze(-1)            
            grad_bias[:, ci] = -gradb

            gradm = myms[:,0,:] * grad_output[:, ci].unsqueeze(-1)
            tmask[:,ci,:] = gradm

            gradi = myis[:,0,:] * grad_output[:, ci].unsqueeze(-1)
            tinput[:,ci,:] = gradi

            # Select input
            #lii = lici[:, ci].unsqueeze(-1)
            #mysel = torch.gather(ssi, 1, lii)
            #tsel[:,ci,0] = mysel.squeeze()
            #tmask[:,ci,0] = grad_output[:, ci]
            #tinput[:,ci,0] = grad_output[:, ci]

        
        grad_weight.scatter_(2, si, tweight)
        grad_mask.scatter_(2, si, tmask)
        grad_input.scatter_(2, si, tinput)

        return grad_input, grad_mask, grad_weight, grad_bias
        

class WOS(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, do_padding=False, test_init=False):
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
        MD = 2*D

        if test_init:
            tempw = torch.zeros(out_channels, MD)
            print("temp inti")
            print(tempw.shape)
            tempw[0, 0] = 1
            tempw[0, 1] = 1
            tempw[0, 2] = 1
            tempw[0, 3] = 1
            #tempw[0, 4] = 0
            #tempw[0, 5] = 1.1
            #if out_channels > 1:
            #    tempw[1, 0] = 1 
            #    tempw[1, 2] = 1.1
            #    tempw[1, 3] = 0.5
            #    tempw[1, 4] = 0.0 
            #    tempw[1, 5] = 0.0

            self.weight = nn.Parameter(tempw, requires_grad=True)
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
            nn.init.constant_(self.bias, 2) 

            tempm = torch.zeros(out_channels, MD)
            tempm[0, 0] = 0
            tempm[0, 1] = 0
            tempm[0, 2] = 1
            tempm[0, 3] = 1

            self.mask = nn.Parameter(tempm, requires_grad=True)
            #self.mask = nn.Parameter(torch.Tensor(out_channels, MD), requires_grad=True)
            #nn.init.zeros_(self.mask)          
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channels, MD), requires_grad=True)
            #nn.init.ones_(self.weight)
            nn.init.uniform_(self.weight, 0.0, 1.0)

            self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
            #nn.init.zeros_(self.bias)
            #temp = self.weight.sum() / 2.0
            #nn.init.constant_(self.bias, temp)
            temp = self.weight.sum(1)
            for i in range(out_channels):
                nn.init.uniform_(self.bias[i], 0.0, temp[i].item())

            self.mask = nn.Parameter(torch.Tensor(out_channels, MD), requires_grad=True)
            nn.init.uniform_(self.mask, -1.0, 1.0)
            #nn.init.zeros_(self.mask)

        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
        self.padding = do_padding
        self.verbose = False

        self.gradWs = torch.zeros((MD, MD))
        self.gradMs = torch.zeros((MD, MD))
        self.gradIs = torch.zeros((MD, MD))
        
        #stepVal = torch.cat( (-torch.ones((D)), torch.ones((D))), 0)
        #stepVal = torch.cat((torch.arange(-D, 0), torch.arange(1, D+1)))
        #print(stepVal)
        for li in range(MD):
            #self.gradWs[li, :] = stepVal
            #self.gradWs[li, :] = torch.cat( (-torch.ones((li+1)), torch.ones((MD-li-1))), 0)        # Asymmetric step
            #self.gradWs[li, :] = torch.cat((torch.arange(-(li), 1), torch.arange(1, (MD-li))))     # Symmetric hinge
            self.gradWs[li, :] = torch.cat((torch.arange(-(li+1), 0), torch.arange(1, (MD-li))))    # Asymmetric hinge
            
            self.gradMs[li, li] = 1.0
            #self.gradMs[li, :] = torch.cat((torch.arange(MD-li, MD+1), torch.arange(MD-1, li, -1)))

            self.gradIs[li, li] = 1.0
            #self.gradIs[li, :] = self.gradMs[li, :]            
            #self.gradIs[li, :] = torch.cat((torch.arange(MD-li, MD+1), torch.arange(MD-1, li, -1)))
        
        #print(self.gradWs)
        OHMWosFunction.gradWs[MD] = self.gradWs
        OHMWosFunction.gradMs[MD] = self.gradMs
        OHMWosFunction.gradIs[MD] = self.gradIs
        
        #print("===========> Assigning ")
        #print(OHMWosFunction.gradWs.shape)

        
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
    #    print("   CALL TO BACKWARD ")
    #    super(WOS, self).backward(gradout)
        

    def MyPrint(self):
        print("Weight: " + str(self.weight.data))
        print("Bias  : " + str(self.bias.data))
        print("Mask  : " + str(self.mask.data))

    def MyShapes(self):
        print("Weight: " + str(self.weight.shape))
        print("Bias  : " + str(self.bias.shape))
        print("Mask  : " + str(self.mask.shape))

    def MyStats(self):
        nw = self.weight.detach().cpu().numpy().squeeze()
        nb = self.bias.detach().cpu().numpy().squeeze()
        #print(nw.shape)
        #print(nb.shape)
        #print('.', end='', flush=True)
        if len(nw.shape) > 1:
            for j in range(nw.shape[0]):
                for i in range(nw.shape[1]):
                    print(str(nw[j,i]) + " ", end='')
            print("")
        elif len(nw.shape) > 0:
            for i in range(nw.shape[0]):
                print(str(nw[i]) + " ", end='')
            print("")
        else:
            print(str(nw))
        
        if len(nb.shape) > 0:
            for i in range(nb.shape[0]):
                print(str(nb[i]) + " ", end='')
            print("")
        else:
            print(str(nb))



    def FindRanks(self, xx, rr):        
        x = xx.clone()
        result = rr.clone()

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

        bits = WOS_PRECISION
        dynRange = 2**(bits)
        #print("Find ranks for " + str(bits) + " bits and range 0 -> " + str(dynRange-1))        
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
        #print(dmx.shape)
        minmax = torch.tensor([np.min(dmx), np.max(dmx)])
        nmx = (dynRange - 1.0) * (mx - minmax[0]) / (minmax[1] - minmax[0])
        #print("Normalized input")
        #print(nmx)

        nresult = (dynRange - 1.0) * (result - minmax[0]) / (minmax[1] - minmax[0])
        
        #print("Normalized result")
        #print(nresult) 
        
        clocks = torch.zeros((nmx.shape[0], NC))
        for ci in range(NC):
            #print(f"Chanel {ci} of {NC}") 
            w = lweight[:,ci,:].detach().numpy().squeeze()
            data = nmx[:,ci,:].squeeze()
            #print(data.shape)
            rest = nresult[:,ci].squeeze()
            #print(rest.shape)
            if len(rest.shape) > 2:
                res = rest.reshape((rest.shape[0]*rest.shape[1]*rest.shape[2]))
            else:
                res = rest
            
            #print(res.shape)
            for n in range(data.shape[0]):
                #print("################################")
                #print(data[n])
                #print(res[n])
                refs, dirs = GetRefs(res[n], bits)                
                notOutput = np.zeros((D))
                notOutput[w < ZERO_TOL] = True
                numNotOutput = sum(notOutput)
                if numNotOutput >= D-1:
                    bi = 0                    
                else:
                    for bi in range(len(refs)):
                        # check all inputs against ref
                        #print("Checking against ref " + str(refs[bi]))
                        for di in range(D):
                            if notOutput[di] == False:
                                if data[n,di] >= refs[bi]:
                                    if dirs[bi] == False:
                                        notOutput[di] = True
                                else:
                                    if dirs[bi] == True:                                        
                                        notOutput[di] = True
                                
                        numNotOutput = sum(notOutput)
                        if numNotOutput >= D-1:
                            break
                    bi = bi + 1
                #print("Sample " + str(n) + ":" + str(numNotOutput) + " at " + str(bi))
                clocks[n, ci] = bi
        
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
    print('HELLO') 
    refs = GetRefs(255, 8)
    print(refs)
