import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from difflogic import LogicLayer, GroupSum, DL_FUNCTIONS

class DiffLogicPBF(nn.Module):

    def __init__(self, num_neurons=4, num_layers=2, connections = 'random'):

        super(DiffLogicPBF, self).__init__()
        
        in_dim = 2 
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
        tx = (x > 0.0).to(torch.float32)        
        ret  = self.model(tx)
        #ret  = self.model(x)
        return ret
    
    def extra_repr(self):
        lfns = ''
        for i, layer in enumerate(self.model):
            if isinstance(layer, LogicLayer):
                
                tweights = torch.nn.functional.one_hot(layer.weights.argmax(-1), 16).to(torch.float32)
                indices = tweights.argmax(dim=1).tolist()
                functions = [DL_FUNCTIONS[index] for index in indices]
                lfns += f'Layer {i}: {functions}\n'

        return lfns    