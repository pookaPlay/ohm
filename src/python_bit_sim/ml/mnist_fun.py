import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# Function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(img.numpy().squeeze(), cmap='gray')
    plt.show()


def TryIt():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

        
    # Display the first image from the batch
    print(images.shape)
    imshow(images[0])

    # Download and load the test data
    #testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Get one batch of images from the trainloader


if __name__ == "__main__":
    TryIt()    
    print("Done")
