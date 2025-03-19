import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from difflogic_ptf import LogicLayer, GroupSum, GetFunctionText, GetNumFunctions
import random

class PTFLogicClassifier(nn.Module):

    def __init__(self, param):

        super(PTFLogicClassifier, self).__init__()
        
        in_dim = 2 
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
        return self.model(x)    
    
    def extra_repr(self):
        lfns = ''
        for i, layer in enumerate(self.model):
            if isinstance(layer, LogicLayer):                
                numFuncs = GetNumFunctions(layer.fan_in)
                print(f'At layer {i} we have numFuncs: {numFuncs}')
                tweights = torch.nn.functional.one_hot(layer.weights.argmax(-1), numFuncs).to(torch.float32)
                indices = tweights.argmax(dim=1).tolist()
                functions = [GetFunctionText(layer.fan_in, index) for index in indices]
                lfns += f'Layer {i}: {functions}\n'
                #lfns += f' {len(layer.indices)} ind: {layer.indices}\n'

        return lfns        
