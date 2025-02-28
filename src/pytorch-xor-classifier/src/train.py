import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import generate_xor_data, generate_linear_data

# Define a simple linear classifier
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(2, 2, bias=True)  # Ensure bias term is included

    def forward(self, x):
        return self.linear(x)


# Define a simple linear classifier
class MorphClassifier(nn.Module):
    def __init__(self):
        super(MorphClassifier, self).__init__()
        self.linear = nn.Linear(2, 2, bias=True)  # Ensure bias term is included

    def forward(self, x):
        return self.linear(x)

# Train the model
def train_model(model, data, labels, optimizer, num_epochs):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.HingeEmbeddingLoss(margin=1.0)
    
    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs = model(data)
        outputs = outputs[:, 1] - outputs[:, 0]  # Convert to binary classification
        loss = criterion(outputs, 2 * labels - 1)  # Convert labels to -1 and 1
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            visualize_decision_surface(model, data, labels)

# Visualize the decision surface
def visualize_decision_surface(model, data, labels):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid)
    Z = Z.argmax(dim=1).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors='k', marker='o')
    plt.show()

if __name__ == "__main__":
    # Hyperparameters
    num_samples = 1000
    num_epochs = 200
    learning_rate = 0.01

    # Generate data
    #data, labels = generate_xor_data(num_samples)    
    data, labels = generate_linear_data(num_samples)
    
    # Initialize model, optimizer
    model = LinearClassifier()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, data, labels, optimizer, num_epochs)