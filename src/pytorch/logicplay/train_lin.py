import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from DataIO import SerializeMSBOffset, DeserializeMSBOffset, generate_xor_data, generate_linear_data

import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_MAX = 128

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
def train_model(model, dataloader, optimizer, num_epochs, viz_epoch):
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

            if 0:
                print("Gradients for epoch {}: ".format(epoch + 1))
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad}")

            optimizer.step()

            if 0:
                print("Parameter values after update  {}: ".format(epoch + 1))
                for name, param in model.named_parameters():
                    print(f"{name}: {param.data}")
        
        model.eval()
        if (epoch + 1) % viz_epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            visualize_decision_surface(model, data, labels, ax)
            plt.pause(0.1)  # Pause to update the figure

   
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Visualize the decision surface
def visualize_decision_surface(model, data, labels, ax):
    #x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    #y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x_min = -DATA_MAX
    x_max = DATA_MAX
    y_min = -DATA_MAX
    y_max = DATA_MAX

    GRID_SPACE = 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, GRID_SPACE),
                         np.arange(y_min, y_max, GRID_SPACE))
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
    num_samples = 100
    batch_size = 10
    num_epochs = 100   
    viz_epoch = 1
    # linear 
    learning_rate = 0.0001
    #learning_rate = 0.1
        
    # Generate data
    data, labels = generate_linear_data(num_samples)
    #data, labels = generate_xor_data(num_samples)

    data = data.float()
    labels = labels.float()
        
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LinearClassifier()
    #model = MorphClassifier()
    model.eval()

    if 0:
        print("Parameter values init")
        print(f'Biases: {model.biases}')        
        print(f'Weights: {model.weights}')
        print(f'Threshold: {model.threshold}')
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    visualize_decision_surface(model, data, labels, ax)
    plt.ioff()  # Turn off interactive mode
    plt.show()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    
    train_model(model, dataloader, optimizer, num_epochs, viz_epoch)

    if 1:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        
        visualize_decision_surface(model, data, labels, ax)
        plt.ioff()  # Turn off interactive mode
        plt.show()
        
        if 0:
            print("Parameter values final")
            print(f'Biases: {model.biases}')        
            print(f'Weights: {model.weights}')
            print(f'Threshold: {model.threshold}')
