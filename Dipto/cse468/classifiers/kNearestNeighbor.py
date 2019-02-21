import numpy as np
from past.builtins import xrange

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        """
        Train the classifier or just memorizing the training data.
        """
        
        self.X_train = X
        self.Y_train = y
        
        
    def predict(self, X, k=1):
        """
        Predict labes for test data using this classifier
        """
        
        dists = self.compute_distances_no_loops(X)
        
        return self.predict_labels(dists, k=k)
    
    
    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train
        L2 = sqrt(X**2 - X_train ** 2)
        """
        X_squared = np.sum(X**2,axis=1)
        Y_squared = np.sum(self.X_train**2,axis=1)
        XY = np.dot(X, self.X_train.T)
        
        # Expand L2 distance formula to get L2(X,Y) = sqrt((X-Y)^2) = sqrt(X^2 + Y^2 -2XY)
        dists = np.sqrt(X_squared[:,np.newaxis] + Y_squared -2*XY)
        return dists

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            k_nearest_index = np.argsort(dists[i, :])[:k]

            closest_y = self.y_train[k_nearest_index]
            labels_counts = {}
            for label in closest_y:
                if label in labels_counts.keys():
                    labels_counts[label] += 1
                else:
                    labels_counts[label] = 0
            sorted_labels_counts = sorted(
                labels_counts.items(), key=operator.itemgetter(1), reverse=True)
            y_pred[i] = sorted_labels_counts[0][0]
        return y_pred
                
        