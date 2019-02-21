import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Naive_Euclid_Classifer import Naive_Euclid
import Array_Loader
import Metrics_Plots as mp
import ColorHue_Histogram
import OrientedGradients_Histogram


#load up arrays of 1d images and its labels from npz files
imgarr, lbarr  = Array_Loader.loadCIFARfromNPZasImageArray()
#X = np.reshape(X, (len(X), 3072))
testimg, testlab = Array_Loader.extractBatchFiles("cifar10/", "test_batch")

# reshape vectors to accomodate function argument
Zt = np.reshape(testimg, (10000,3,32,32))
Z1 = np.moveaxis(Zt,1,3)
Xt = np.moveaxis(imgarr,1,3)

# extract features using various algorithms
print(Xt.shape,Z1.shape)
print(type(Xt),type(Z1))
X1 = []
Z2 = []
ind=0
for i in Xt:
    X1.append(OrientedGradients_Histogram.generate_Hog_Descriptor(i, (4,4)))
    ind += 1
    if ind%10==0:print('Hog extracted:',ind)
ind=0
for i in Z1:
    Z2.append(OrientedGradients_Histogram.generate_Hog_Descriptor(i, (4,4)))
    ind += 1
    if ind%10==0:print('Hog extracted:',ind)
X = np.array(X1)
Z = np.array(Z2)

# prepare model
model = Naive_Euclid()
d = {'label': lbarr}
Y = pd.DataFrame(data=d)

# predict labels using euclid model
model.fit(X, Y)
pred_lbls = model.predictMultiple(Z)

# save and load predicted labels as npy for later use
np.save('PredLabels_HOG',pred_lbls)
# pred_lbls = np.load('PredLabels_HOG.npy')

# compute confusion matrix, accuracy and plot
confmat, accuracy = mp.computeConfMat(pred_lbls, testlab, norm=True)
mp.plotConfMat(confmat, title='CM HOG')
print(accuracy)
