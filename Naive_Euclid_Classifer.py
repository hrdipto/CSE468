import pandas as pd
import numpy as np

class Naive_Euclid(object):

    def __init__(self, k_neighbors=1):

        #expected 2d array where each row is a 1d array containing all pixels or
        #an image. (Each row therefore is an 1d representation of an image)
        self.dataset = np.asarray([[]])
        self.labels = pd.DataFrame()

        self.k_neighbors = k_neighbors

    #Will ensure X is an numpy array of appropriate shape
    #Will throw error otherwise
    def __validate_dataset(self, X):

        return 0


    def fit(self, X, Y):

        self.__validate_dataset(X)

        self.dataset = X
        self.labels = Y


    def predict(self, X):


        diff_mat = self.dataset - X  #the pixel difference
        squared_diff = diff_mat**2
        summed_diff = squared_diff.sum(axis=1)
        euc_dis = summed_diff**.5

        #coverting to pandas df:
        euc_dis = pd.DataFrame(euc_dis)

        labels_with_distance = self.labels
         #now we have a df containing labels along with another column that contains
         #the distance between that image and the image we are predicting.
        labels_with_distance['euclidean_distance'] = euc_dis

        #sorting dataframe based on eucleadian distance
        labels_with_distance = labels_with_distance.sort_values('euclidean_distance')

        pred_class = labels_with_distance[0:1]['label'].values[0] #getting label of smallest distance


        return pred_class


    def predictMultiple(self, X):

        pred_labels = []
        totali = len(X)
        for i, x_test in enumerate(X):
            #predict the label from image of different batch using model
            pred = self.predict(x_test)
            pred_labels.append(pred)
            if i%10==0: print('Predict Element no.',i,'of',totali)

        return pred_labels
