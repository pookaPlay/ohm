import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
#import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.decomposition import PCA
#import SegEval as ev
#import SynGraph as syn
#from PIL import Image
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.io import imread
from scipy.io import loadmat
#import os
#from skimage.viewer import ImageViewer
import matplotlib.image as mpimg
import networkx as nx

TRAIN_GT_DIR = '../../data/BSR_bsds500/BSR/BSDS500/data/groundTruth/train/'
TRAIN_IMG_DIR = '../../data/BSR_bsds500/BSR/BSDS500/data/images/train/'

TRAIN_GT_EXT = '.mat'
TRAIN_IMG_EXT = '.jpg'

TRAIN_NAME = ['277095']
#TRAIN_NAME = ['100075']
#TRAIN_NAME = ['326038']


def LoadTrain(id = 0):
    
    idName = TRAIN_NAME[id]
    
    gtFile = TRAIN_GT_DIR + idName + TRAIN_GT_EXT
    imgFile = TRAIN_IMG_DIR + idName + TRAIN_IMG_EXT
    
    
    gts = loadmat(gtFile)
    gt = gts['groundTruth']
    num_gts = gt.shape[1]

    print(gt.shape)
    segId = 0
    seg = gt[0,segId]['Segmentation'][0,0].astype(np.float32)
    print(seg.shape)
 
    img = img_as_float(imread(imgFile))
    print(img.shape)

    return(img, seg)

def ScaleAndCropData(imgf, segf):
    #print(imgf.shape)

    img = imgf[::2, ::2,:]
    seg = segf[::2, ::2]
    img1 = img
    img2 = img
    seg1 = seg
    seg2 = seg
    
    img1 = img[100:164, 60:124, :]    
    seg1 = seg[100:164, 60:124]    
    img2 = img[100:164, 4:68, :]
    seg2 = seg[100:164, 4:68]
    print(img1.shape)
    print(seg1.shape)
    print(img2.shape)
    print(seg2.shape)
    return (img1, seg1, img2, seg2)


def VizTrainTest(img1, seg1, img2, seg2):
    
    fig=plt.figure(figsize=(8, 8))

    fig.add_subplot(2, 2, 1)
    plt.imshow(img1)
    fig.add_subplot(2, 2, 2)
    plt.imshow(seg1)
    fig.add_subplot(2, 2, 3)
    plt.imshow(img2)
    fig.add_subplot(2, 2, 4)
    plt.imshow(seg2)
    
    return    

        
if __name__ == '__main__':    
    print("Loading")
    (img, seg) = LoadTrain(0)    
    (img1, seg1, img2, seg2) = ScaleAndCropData(img, seg)

    VizTrainTest(img1, seg1, img2, seg2)
    plt.show()

    print("Done")
