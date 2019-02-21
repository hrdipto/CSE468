import numpy as np
import _pickle as pickle
import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt
import glob

# label_nums:::  0          1               2     3       4     5       6       7       8       9
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#read and extract (pickle) the image array data from batch files
def extractBatchFiles(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='latin1')
    images = dict['data']
    labels = dict['labels']
    imagearray = np.array(images)
    labelarray = np.array(labels)
    return imagearray, labelarray

#read png image files from cifar10/train folder
def readImageFiles(path):
    imagearray = []
    for filename in glob.glob(path + 'train/' + '*.png'): #assuming gif
        img = imread(filename)
        img = img.flatten()
        imagearray.append(img)

    img = imread(path+'1.png')
    print(type(imagearray), img.shape)
    img = img.flatten()
    print(type(img), img.shape)

# def reshapeDisplayImagefromArray(img1d):
#     img = np.reshape(img1d, (3, 32, 32))
#     #img = np.moveaxis(img, 0, 2)
#     print(img.shape)
#     plt.imshow(img)
#     plt.savefig('img.png')

def load1DbatchArrayto3Dnpy():
    imgarr, lbarr  = extractBatchFiles("cifar10/", "data_batch_1")
    for i in range(2,6):
        X_temp, lbarr_temp  = extractBatchFiles("cifar10/", "data_batch_"+str(i))
        imgarr = np.append(imgarr, X_temp, axis=0)
        lbarr = np.append(lbarr, lbarr_temp, axis=0)
    testimg, testlab = extractBatchFiles("cifar10/", "test_batch")
    newarr = []
    for i in range(0,len(imgarr)):
        newarr.append(np.reshape(imgarr[i], (3, 32, 32)))
    # indexx = 6
    # print(label_names[lbarr[indexx]], lbarr[indexx])
    # plt.imshow(np.moveaxis(newarr[indexx],0,-1))
    # plt.show()
    np.save('CIFAR10_ImageArray_5kRGB',newarr)
    np.save('CIFAR10_LabelArray_5k',lbarr)
    return newarr, lbarr

def loadCIFARfromNPZasImageArray():
    npzimg = np.load('CIFAR10_ImageArray_5kRGB.npy')
    npzlbl = np.load('CIFAR10_LabelArray_5k.npy')
    return npzimg, npzlbl
