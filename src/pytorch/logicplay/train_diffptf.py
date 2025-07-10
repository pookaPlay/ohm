import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from DataIO import generate_xor_data, generate_xxor_data, generate_linear_data, generate_3nor_data
import random

from PTFLogicClassifier import PTFLogicClassifier
from DiffLogicClassifier import DiffLogicClassifier

# Train the model
def train_model(model, dataloader, num_epochs, viz_epoch):

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        
        model.eval()

        if (epoch + 1) % viz_epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            #print(f'ping')
            visualize_decision_surface(model, data, labels, ax)
            #print(f'pong')
            plt.pause(0.1)  # Pause to update the figure
                    

    plt.ioff()  # Turn off interactive mode
    plt.show()

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

# Visualize the decision surface
def visualize_decision_surface(model, data, labels, ax):
    DATA_MAX = 1
    x_min = -DATA_MAX
    x_max = DATA_MAX
    y_min = -DATA_MAX
    y_max = DATA_MAX    
    step = 0.1
    #dmin = torch.min(data)
    #dmax = torch.max(data)
    #print(f'dmin: {dmin}, dmax: {dmax} ans step: {step}')

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        
    with torch.no_grad():
        Z = model(grid)
    
    #Z = torch.where(Z >= 0, 1, 0)  # Threshold Z at 0
    #Z = Z.reshape(xx.shape)
    Z = Z.argmax(dim=1).reshape(xx.shape)        
    
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
    
    dlopt = dict(
        num_neurons = 4, 
        num_layers = 4, 
        connections = 'random',
        fan_in = 4,
        grad_factor = 1.0,       
        #connections = 'unique'
        )
    print(dlopt)
    # linear 
    #learning_rate = 0.0001
    learning_rate = 0.01
        
    # Generate data
    #data, labels = generate_linear_data(num_samples)
    #data, labels = generate_xxor_data(num_samples)
    data, labels = generate_xor_data(num_samples)
    #data, labels = generate_3nor_data(num_samples, 3)
    
    # move from +-1 to 0,1
    labels = (labels + 1) / 2
    
    data = data.float()
    labels = labels.long()        
    
    # print the data min and max 
    print(f'data min: {torch.min(data)}, data max: {torch.max(data)}')

    # move from +-1 to 0,1
    #dataOrig = (data + DATA_MAX) / (2. * DATA_MAX)
    #data = dataOrig
        

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    #model = DiffLogicClassifier(dlopt)    
    model = PTFLogicClassifier(dlopt)
    
    model.eval()
    print(model)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    visualize_decision_surface(model, data, labels, ax)
    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Train the model
    #print(model)

    train_model(model, dataloader, num_epochs, viz_epoch)

    print(model)

    if 1:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        
        visualize_decision_surface(model, data, labels, ax)
        plt.ioff()  # Turn off interactive mode
        plt.show()
