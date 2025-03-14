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

STACK_BITS = 8
DATA_MAX = 128

class CustomWOS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, threshold):
        
        N = x.shape[0]
        D = x.shape[1]
        
        input_bits = torch.zeros(N, D, STACK_BITS)
        sticky_bits = torch.zeros(N, D)
        output_bits = torch.zeros(N, STACK_BITS)
        output = torch.zeros(N)
        outputIndex = torch.zeros(N, dtype=torch.int32)
        posCount = torch.zeros(N)        
        
        #print('##########################################')
        #print(f' Start of call: {weights} -> {threshold}')
        for ni in range(N):  
            for di in range(D):  
                val = SerializeMSBOffset(x[ni,di].item(), STACK_BITS)
                input_bits[ni,di,:] = torch.tensor(val, dtype=torch.float32)
                
            input_values = input_bits[ni,:,0]
            for k in range(STACK_BITS):        
                for di in range(D):            
                    if sticky_bits[ni,di] == 0:
                        input_values[di] = input_bits[ni,di,k]
                                
                weighted_sum = sum(w for i, w in zip(input_values, weights) if i > 0)
                if weighted_sum >= threshold:     
                    output_bits[ni,k] = 1    
                else: 
                    output_bits[ni,k] = 0                

                for di in range(D):            
                    if sticky_bits[ni,di] == 0:
                        if output_bits[ni,k] != input_bits[ni,di,k]:
                            sticky_bits[ni,di] = k + 1
            
            out_index = (sticky_bits[ni, :] == 0).nonzero(as_tuple=True)[0]
            assert len(out_index) > 0, f"Expected at least 1 zero sticky index, got {len(out_index)}"
            oi = out_index[0]
            outputIndex[ni] = oi
            output[ni] = DeserializeMSBOffset(output_bits[ni,:].tolist())

            if 0:
                posCount[ni] = torch.sum(output_bits[ni,:]) / STACK_BITS
                MAX_THRESH = torch.sum(weights)
                if posCount[ni] < 0.5:
                    # decrease threshold
                    threshold = threshold - 1.
                    if threshold < 1:
                        threshold = torch.tensor([1.])
                        # bump weight
                        weights[oi] = weights[oi] + 1.
                else: 
                    # increase threshold
                    threshold = threshold + 1.
                    if threshold > MAX_THRESH:
                        threshold = torch.tensor([MAX_THRESH])
                        # bump weight
                        weights[oi] = weights[oi] + 1.
                
        #print(f' After {N}: {weights} -> {threshold}')
        if 0:
            if torch.min(weights) > threshold:
                adj = torch.min(weights) - threshold            
                weights = weights - adj       
                print(f'ADJ: {weights} -> {threshold}')                 

        ctx.save_for_backward(outputIndex, x, weights, threshold)        
        
        return(output)

    @staticmethod
    def backward(ctx, grad_output):
                
        outputIndex, x, weights, threshold = ctx.saved_tensors
        N = x.shape[0]
        D = x.shape[1]
                
        grad_x = torch.zeros([x.shape[0], x.shape[1]])
        grad_weights = torch.zeros([weights.shape[0]])
        grad_threshold = torch.zeros([threshold.shape[0]])
        
        for ni in range(N):
            grad_x[ni, outputIndex[ni]] = grad_output[ni] * 100.0
                
        return grad_x, grad_weights, grad_threshold


class MorphClassifier(nn.Module):

    def __init__(self):
        super(MorphClassifier, self).__init__()
        D = 2
        D2 = D*2
        
        biasInit = torch.tensor([-64., -64., -64., -64.])        
        threshInit = torch.tensor([2.])

        self.biases = nn.Parameter(biasInit)        
        self.threshold = nn.Parameter(threshInit)
        self.weights = nn.Parameter(torch.ones(D2))
        
    def forward(self, x):
        concatenated_result = torch.cat((x, -x), dim=1)
        
        lsb_result = concatenated_result + self.biases
                
        msb_result = CustomWOS.apply(lsb_result, self.weights, self.threshold)
        
        return msb_result        

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
        
    model = MorphClassifier()
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
