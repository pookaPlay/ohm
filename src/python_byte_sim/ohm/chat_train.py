import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from chat_data import generate_xor_data, generate_linear_data
from torch.utils.data import DataLoader, TensorDataset
from DataIO import vector_to_twos_complement, twos_complement_to_int

import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define a custom autograd function for a linear classifier
class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, bias):
        ctx.save_for_backward(x, weights, bias)
        return torch.matmul(x, weights) + bias

    @staticmethod
    def backward(ctx, grad_output):        
        x, weights, bias = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, weights.t())
        grad_weights = torch.matmul(x.t(), grad_output)
        grad_bias = grad_output.sum(0)
        return grad_x, grad_weights, grad_bias
    
# Define a simple linear classifier without using nn.Linear
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.inD = 2
        self.outD = 1
        self.weights = nn.Parameter(torch.randn(self.inD, self.outD))  # Initialize weights
        self.bias = nn.Parameter(torch.randn(self.outD))  # Initialize bias        

    def forward(self, x):        
        return CustomLinear.apply(x, self.weights, self.bias)        

# Define a custom autograd function
class CustomMorph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, biases):
        lsb_result = x + biases
        msb_result, msb_index = torch.max(lsb_result, dim=1)
        ctx.save_for_backward(x, biases, msb_index)
        return msb_result

    @staticmethod
    def backward(ctx, grad_output):
        x, biases, msb_index = ctx.saved_tensors
        
        grad_input = torch.zeros(x.shape[0], x.shape[1])
        for n in range(grad_output.shape[0]):
            grad_input[n, msb_index[n]] = grad_output[n]
        
        grad_biases = torch.zeros(biases.shape[0], dtype=grad_output.dtype)
        for n in range(grad_output.shape[0]):            
            grad_biases[msb_index[n]] += grad_output[n]
        #print(f'Grad biases: {grad_biases}')
        return grad_input, grad_biases

STACK_BITS = 8

class CustomWOS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, threshold):
        #ctx.save_for_backward(x, weights, threshold)
        #result = torch.matmul(x, weights) + threshold
        input_bits = torch.zeros(x.shape[0], x.shape[1], STACK_BITS)
        output_bits = torch.zeros(x.shape[0], STACK_BITS)
        output = torch.zeros(x.shape[0])

        for i in range(x.shape[0]):     
            maxval, maxind = torch.max(x[i])
            input_bits[i,:,:] = vector_to_twos_complement(x[i], STACK_BITS)            
            output_bits[i,:] = input_bits[i,maxind,:]

            output[i] = twos_complement_to_int(output_bits[i,:])
        
        return output

class MorphClassifier(nn.Module):
    def __init__(self):
        super(MorphClassifier, self).__init__()
        self.inD = 2
        self.outD = 1
        
        self.biases = nn.Parameter(torch.randn(self.inD))
        self.weights = nn.Parameter(torch.ones(self.inD*2, self.outD))
        self.threshold = nn.Parameter(torch.zeros(self.outD))         
        
    def forward(self, x):
        lsb_result = x + self.biases
        concatenated_result = torch.cat((lsb_result, -lsb_result), dim=1)
        #msb_result, msb_index = torch.max(lsb_result, dim=1)
        msb_result = CustomWOS.apply(concatenated_result, self.weights, self.threshold)
        return msb_result
        #return CustomMorph.apply(x, self.biases)

class CustomHinge(nn.Module):
    def __init__(self):
        super(CustomHinge, self).__init__()

    def forward(self, outputs, labels):
        #loss = torch.mean((outputs - labels) ** 2)
        #return loss
        loss = 1.0 - torch.mul(outputs, labels)
        loss[loss < 0.0] = 0.0        
        hingeLoss = torch.mean(loss)      
        return(hingeLoss)

    
# Train the model
def train_model(model, dataloader, optimizer, num_epochs):
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.HingeEmbeddingLoss(margin=1.0)
    criterion = CustomHinge()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    for epoch in range(num_epochs):
        model.train()
        
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            #print(f'outputs: {outputs.shape}')
            #print(f'labels: {labels.shape}')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            visualize_decision_surface(model, data, labels, ax)
            plt.pause(0.1)  # Pause to update the figure

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Visualize the decision surface
def visualize_decision_surface(model, data, labels, ax):
    #x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    #y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x_min = -128
    x_max = 128
    y_min = -128
    y_max = 128
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid)
    
    Z = torch.where(Z >= 0, 1, 0)  # Threshold Z at 0
    Z = Z.reshape(xx.shape)
    #Z = Z.argmax(dim=1).reshape(xx.shape)    
    # viz  at 0
    
    ax.clear()
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(data[:, 0], data[:, 1], c=labels, edgecolors='k', marker='o')
    plt.draw()

if __name__ == "__main__":
    # Hyperparameters
    num_samples = 200
    num_epochs = 1000
    learning_rate = 0.00001
    batch_size = 20

    # Generate data
    data, labels = generate_linear_data(num_samples)
    #data, labels = generate_xor_data(num_samples)

    data = data.float()
    
    print(data)
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LinearClassifier()
    #model = MorphClassifier()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, optimizer, num_epochs)
    
