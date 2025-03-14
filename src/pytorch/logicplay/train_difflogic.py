import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from DataIO import SerializeMSBOffset, DeserializeMSBOffset, generate_xor_data, generate_linear_data
from difflogic import LogicLayer, GroupSum

import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

DATA_MAX = 128

class DiffLogicClassifier(nn.Module):
    def __init__(self):
        super(DiffLogicClassifier, self).__init__()
        
        in_dim = 2 
        class_count = 2
        num_neurons = 16
        num_layers = 2
        tau = 1.
        grad_factor = 1.0
        connections = 'random'
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
        print(f'total_num_neurons={total_num_neurons}')
        total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
        print(f'total_num_weights={total_num_weights}')
        print(self.model)

    def forward(self, x):
        return self.model(x)    
    

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
    DATA_MAX = 128
    GRID_SPACE = 10
    
    x_min = -DATA_MAX
    x_max = DATA_MAX
    y_min = -DATA_MAX
    y_max = DATA_MAX

    
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
    
    model = DiffLogicClassifier()
    
    model.eval()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    visualize_decision_surface(model, data, labels, ax)
    plt.ioff()  # Turn off interactive mode
    plt.show()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #loss_fn = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    
    train_model(model, dataloader, optimizer, num_epochs, viz_epoch)

    if 1:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        
        visualize_decision_surface(model, data, labels, ax)
        plt.ioff()  # Turn off interactive mode
        plt.show()
        