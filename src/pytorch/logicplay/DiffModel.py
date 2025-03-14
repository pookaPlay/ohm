import argparse
import math
import random
import os

import numpy as np
import torch
from difflogic import LogicLayer, GroupSum

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}


def get_model(args):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections)

    in_dim = 2 
    class_count = 2

    logic_layers = []
    k = args.num_neurons
    l = args.num_layers

    ####################################################################################################################

    
    logic_layers.append(torch.nn.Flatten())
    logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
    for _ in range(l - 1):
        logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

    model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(class_count, args.tau)
    )

    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    print(f'total_num_weights={total_num_weights}')
    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer

