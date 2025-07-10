import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
#from difflogic_ptf import LogicLayer, GroupSum, GetFunctionText, GetNumFunctions
from difflogic import LogicLayer, GroupSum, GetFunctionText, GetNumFunctions
import random

class StackLogicClassifier(nn.Module):

    def __init__(self, param):

        super(StackLogicClassifier, self).__init__()
        
        in_dim = 2*2 
        class_count = 2        
        tau = 1.        

        logic_layers = []
        k = param['num_neurons']
        l = param['num_layers']
        
        lparam = dict(
            grad_factor=param['grad_factor'],
            connections=param['connections'],
            fan_in=param['fan_in'],
            )

        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **lparam))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **lparam))

        self.model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, tau)
        )

                
    def forward(self, x):
        N = x.shape[0]
        D = x.shape[1]
        D2 = 2*D

        outD = 2        
        numThresh = 2

        flags = torch.zeros([N, D2, outD])        
        results = torch.zeros([N, numThresh, outD, numThresh])
         
        mx = torch.cat([x, -x], dim=1)        
        input_values = torch.zeros_like(mx)
        #print(f'mx: {mx.shape}')
        thresh = 0.0
        for ti in range(numThresh):
                                    
            tx = (mx > thresh).to(torch.float32)

            for ni in range(N):            
                for oi in range(outD):
                    for di in range(D2):            
                        if flags[ni, di, oi] == 0:
                            input_values[ni, di] = tx[ni,di]
            
            rt = self.model(input_values)
            assert rt.shape[1] == outD

            results[:,ti,:] = rt
            
            for ni in range(N):            
                for oi in range(outD):
                    for di in range(D2):
                        if flags[ni, di, oi] == 0:
                            if input_values[ni,di] != results[ni,ti, oi]:
                                flags[ni, di, oi] = 1
        
        return results
    
    def extra_repr(self):
        lfns = ''
        for i, layer in enumerate(self.model):
            if isinstance(layer, LogicLayer):                
                numFuncs = GetNumFunctions(layer.fan_in)                
                tweights = torch.nn.functional.one_hot(layer.weights.argmax(-1), numFuncs).to(torch.float32)
                indices = tweights.argmax(dim=1).tolist()
                functions = [GetFunctionText(layer.fan_in, index) for index in indices]
                lfns += f'Layer {i}({layer.fan_in}): ({numFuncs}){functions}\n'
                #lfns += f' {len(layer.indices)} ind: {layer.indices}\n'

        return lfns        
