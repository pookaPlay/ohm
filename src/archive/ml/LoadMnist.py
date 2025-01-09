import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

# Function to un-normalize and display an image
def ShowMNIST(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(img.numpy().squeeze(), cmap='gray')
    plt.show()


def LoadSomeData():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32)),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    # Download and load the test data
    #testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    total_images = len(trainset)
    print("Total number of images in the training set:", total_images)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    return images, labels            


def ProcessDataset():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32)),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    # Download and load the test data
    #testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    total_images = len(trainset)
    D = 1024

    # max
    W = torch.ones([D, D]) * smallest_int
    
    # min
    M = torch.ones([D, D]) * largest_int

    print("Total number of images in the training set:", total_images)        
    for images, labels in tqdm(trainloader, total=len(trainloader)):    
        images = images.view(images.shape[0], -1)
        for ii in range(images.shape[0]):
            img = images[ii]
            imgMat = img.view(D, 1) - img.view(1, D)            
            W = torch.max(W, imgMat)
            M = torch.min(M, imgMat)
            
    #torch.save(W, 'W.pt')
    #torch.save(M, 'M.pt')
    return W, M


def TestDataset():
    W = torch.load('W.pt')
    M = torch.load('M.pt')
    print(torch.max(W))
    print(torch.min(W))
    #print(M)
    print(W.shape)
    #print(M.shape)
    imgData, imgLabels = LoadSomeData()
    print(f"Data: {imgData.shape} Labels: {imgLabels.shape}")
    ShowMNIST(imgData[0])
    


if __name__ == "__main__":
    #LoadSomeData()    
    #ProcessDataset()    
    TestDataset()
    print("Done")
