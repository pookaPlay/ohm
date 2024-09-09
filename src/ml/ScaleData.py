from bls.OHM_NETWORK import OHM_NETWORK
from bls.OHM_PROBE import OHM_PROBE
import matplotlib.pyplot as plt
import math
import pickle
import torch
import sys
import random

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

def ScaleData(data, maxValOut, clipValOut) -> None:
    #self.min_value = torch.min(data)
    #self.max_value = torch.max(data)        
    
    minScale = -3.0
    maxScale = 3.0
    #print(f"Scaling from: {self.min_value}->{self.max_value} to {self.minScale}->{self.maxScale}")
    
    # scale 0 -> 1
    data = (data - minScale) / (maxScale - minScale)
    data[data < 0.0] = 0.0
    data[data > 1.0] = 1.0
    # scale -1 -> 1
    data = (data - 0.5) * 2.0
    
    data = data * maxValOut
    
    data[data > clipValOut] = clipValOut
    data[data < -clipValOut] = -clipValOut
            
    data = torch.round(data)
    data = data.int()        
    # take out zeros!
    data = torch.where(data == 0, torch.tensor(1, dtype=data.dtype), data)

    return data

def ReverseScaleData(data, maxValOut, clipValOut) -> None:    
    
    minScale = -3.0
    maxScale = 3.0
    # maxValOut = 127
    data = (data / maxValOut) * maxScale        

    return data
