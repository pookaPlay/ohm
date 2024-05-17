import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
import pickle

# functions to show an image
def ShowCFAR(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def LoadSomeData():

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')    

    images = dict()
    useClasses = ['plane', 'car']


    if True:
        with open(f'plane_car.pkl', 'rb') as f:
            images = pickle.load(f)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=0)
        
        print("Total number of images in the training set:", len(trainset))

        for cls in useClasses:
            images[cls] = []
            cls_label = classes.index(cls)

            for image, label in trainset:
                if label == cls_label:
                    images[cls].append(image)

        print("Saving images")
        with open(f'plane_car.pkl', 'wb') as f:
            pickle.dump(images, f)
    

    imageSet = []
    for cls in useClasses:
        for i in range(5):
            imageSet.append(images[cls][i])

    print("Showing images")
    ShowCFAR(torchvision.utils.make_grid(imageSet, 5))


    print("Done loading data")

LoadSomeData()
