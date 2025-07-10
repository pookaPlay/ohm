import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pdifflogic import LogicLayer, GroupSum, DL_FUNCTIONS

class StackLogicClassifier(nn.Module):

    def __init__(self, num_neurons=4, num_layers=2, connections = 'random'):

        super(StackLogicClassifier, self).__init__()
        
        in_dim = 4 
        class_count = 2
        tau = 1.
        grad_factor = 1.0
        
        llkw = dict(grad_factor=grad_factor, connections=connections)

        logic_layers = []
        k = num_neurons
        l = num_layers

        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

        self.model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, tau)
        )
                
        total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
        #print(f'total_num_neurons={total_num_neurons}')
        total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
        #print(f'total_num_weights={total_num_weights}')        

    def forward(self, x):   
        N = x.shape[0]
        D = x.shape[1]        
        D2 = D * 2
        
        mx = torch.cat([x, -x], dim=1)        
        
        outD = 2        
        numThresh = 2
        threshSpace = 0.5

        flags = torch.zeros([N, D2, outD])        
        results = torch.zeros([N, numThresh, outD])
        thresh = torch.zeros([N])
                
        input_values = torch.zeros_like(mx)                        
        
        for ti in range(numThresh):
            if ti == 0:
                thresh = torch.zeros([N])                
            else:                             
                lastr = results[:,ti-1,:]
                # find the class label
                #print(f'lastr:{lastr.shape}')
                lastl = lastr.argmax(dim=1)
                #print(f'lastl:{lastl.shape}')
                wadj = torch.where(lastl > 0.5, threshSpace, -threshSpace)
                #print(f'wadj:{wadj.shape}')
                thresh = thresh + wadj
                threshSpace = threshSpace / 2.0

            rthresh = thresh.unsqueeze(1).repeat(1, mx.shape[1])

            tx = (mx > rthresh).to(torch.float32)            
            #tx = x

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
                        if input_values[ni,di] != results[ni,ti, oi]:
                            flags[ni, di, oi] = 1


        return results[:,numThresh-1,:]    
    
    def extra_repr(self):
        lfns = ''
        for i, layer in enumerate(self.model):
            if isinstance(layer, LogicLayer):
                
                tweights = torch.nn.functional.one_hot(layer.weights.argmax(-1), 16).to(torch.float32)
                indices = tweights.argmax(dim=1).tolist()
                functions = [DL_FUNCTIONS[index] for index in indices]
                lfns += f'Layer {i}: {functions}\n'

        return lfns    
    