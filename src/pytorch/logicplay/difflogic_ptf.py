import torch
import numpy as np
from functional_ptf import multi_op_s, get_unique_connections, GradFactor, GetFunctionText, GetNumFunctions


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            fan_in: int = 2,
            device: str = 'cpu',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()                
        self.in_dim = in_dim
        self.fan_in = fan_in
        if self.fan_in > self.in_dim:
            self.fan_in = self.in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, GetNumFunctions(self.fan_in), device=device))

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = 'python'
        self.connections = connections        
        assert self.connections in ['random', 'unique'], self.connections

        self.indices = self.get_connections(self.connections, self.fan_in, device)
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)

        return self.forward_python(x)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        inputs = []
        for i in range(len(self.indices)):      
            
            #if self.indices[i].dtype == torch.int64:
            #    self.indices[i] = self.indices[i].long()
            a = x[..., self.indices[i]]
            inputs.append(a)
        
        numFunctions = GetNumFunctions(self.fan_in)

        if self.training:               
            x = multi_op_s(inputs, torch.nn.functional.softmax(self.weights, dim=-1))
        else:            
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), numFunctions).to(torch.float32)
            x = multi_op_s(inputs, weights)
        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, fan_in=2, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        
        if connections == 'random':
            
            c = torch.zeros((self.fan_in, self.out_dim), dtype=torch.int64, device=device)

            for j in range(self.out_dim):
                # i want c[:,j] to be a permutation of the in_dim
                c[:, j] = torch.randperm(self.in_dim)[:self.fan_in]
            
            print(f'c: {c}')    
            return c
        
        if connections == 'random2':
            c = torch.randperm(self.fan_in * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(self.fan_in, self.out_dim)                    

            for i in range(self.fan_in):
                c[i] = c[i].to(torch.int64)            
                c[i] = c[i].to(device)

            print(f'c: {c}')                
            return c
        
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cpu'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)

