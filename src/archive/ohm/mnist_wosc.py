from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from wos import ClipNegatives
from wos import WOS
import random
#from linear import Linear

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
#NO_CUDA = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #print(device)
        self.conv1 = WOS(1, 16, 3)
        self.conv2 = WOS(16, 64, 3)
        self.fc1 = WOS(9216, 32, 1)
        self.fc2 = WOS(32, 10, 1)

        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.fc1 = nn.Conv2d(9216, 128, 1, 1)
        #self.fc2 = nn.Conv2d(128, 10, 1, 1)

        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print("FWD")
        #print(x.shape)
        self.conv1 = self.conv1.to(device)
        x = self.conv1(x)        
        #print(x.shape)
        #x = torch.tanh(x)        
        self.conv2 = self.conv2.to(device)
        
        x = self.conv2(x)
        #print(x.shape)
        #x = torch.tanh(x)
        x = F.max_pool2d(x, 2)        
        x = torch.flatten(x, 1)        
        x = x.unsqueeze(-1)        
        x = x.unsqueeze(-1)
        #print(x.shape)
        self.fc1 = self.fc1.to(device)
        x = self.fc1(x)
        #print(x.shape)
        #x = torch.tanh(x)
        self.fc2 = self.fc2.to(device)        
        x = self.fc2(x)        
        #print(x.shape)
        x = x.squeeze()        
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, postop, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.to(device)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.apply(postop) 
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, bestError):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    error = (len(test_loader.dataset)-correct) / len(test_loader.dataset)
    if error < bestError:
        bestError = error
        torch.save(model.state_dict(), "best_ohm_mnist.pth")

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return bestError


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device = "cpu" 

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    #print(device)
    model = Net().to(device)
    clipper = ClipNegatives()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    bestError = 1.0
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, clipper, epoch)
        bestError = test(model, device, test_loader, bestError)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_wosc.pt")


if __name__ == '__main__':
    main()
