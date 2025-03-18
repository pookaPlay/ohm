import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from DataIO import generate_xor_data, generate_linear_data, generate_3nor_data
#from difflogic_pbf_mirrored import LogicLayer, GroupSum, DL_FUNCTIONS
from difflogic_ptf import LogicLayer, GroupSum, DL_FUNCTIONS
#from difflogic import LogicLayer, GroupSum, DL_FUNCTIONS
import random

class PTFLogicClassifier(nn.Module):

    def __init__(self, num_neurons=4, num_layers=2, connections = 'random'):

        super(PTFLogicClassifier, self).__init__()
        
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
        print(f'total_num_neurons={total_num_neurons}')
        total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
        print(f'total_num_weights={total_num_weights}')        

    def forward(self, x):
        return self.model(x)    
    
    def extra_repr(self):
        lfns = ''
        for i, layer in enumerate(self.model):
            if isinstance(layer, LogicLayer):
                
                tweights = torch.nn.functional.one_hot(layer.weights.argmax(-1), 16).to(torch.float32)
                indices = tweights.argmax(dim=1).tolist()
                functions = [DL_FUNCTIONS[index] for index in indices]
                lfns += f'Layer {i}: {functions}\n'

        return lfns        

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

# Visualize the decision surface
def visualize_decision_surface(model, data, labels, ax):
    #x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    #y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    #DATA_MAX = 128
    #GRID_SPACE = 10
    DATA_MAX = 1
    GRID_SPACE = 0.1
    
    x_min = -DATA_MAX
    x_max = DATA_MAX
    y_min = -DATA_MAX
    y_max = DATA_MAX

    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, GRID_SPACE),
                         np.arange(y_min, y_max, GRID_SPACE))
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



seed = 2
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

DATA_MAX = 128

if __name__ == "__main__":
    # Hyperparameters    
    num_samples = 100
    batch_size = 10
    num_epochs = 10   
    viz_epoch = 1
    
    dlopt = dict(
        num_neurons = 4, 
        num_layers = 2, 
        connections = 'random'
        #connections = 'unique'
        )
    print(dlopt)
    # linear 
    #learning_rate = 0.0001
    learning_rate = 0.1
        
    # Generate data
    #data, labels = generate_linear_data(num_samples)
    data, labels = generate_xor_data(num_samples)
    # data, labels = generate_3nor_data(num_samples, 3)
    
    # move from +-1 to 0,1
    labels = (labels + 1) / 2
    
    data = data.float()
    data = data / DATA_MAX
    labels = labels.long()    

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PTFLogicClassifier(**dlopt)    
    
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
