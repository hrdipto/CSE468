# import our libraries
from Naive_Euclid_Classifer import Naive_Euclid

# assigned wrapper class
class NaiveClassifier:

    # trainimages [numofimages, numofchannels, xdim, ydim] numpy array
    # trainlabels [numoflabels, 1]
    def __init__(self, trainingImages, trainingLabels, featureExtractor):
        self.trainX = trainingImages
        self.trainY = trainingLabels
        self.extractor = featureExtractor
        self.featureArray = []

    # image [numofchannels, xdim, ydim] numpy array
    def extract_feature_from_single_image(self, image):
        # not sure how this is going to differentiate between the 3 algos
        # by sending a particular featureExtractor object from subclass?
        features = self.extractor.extract_feature(image)
        return features

    # imageArray [numofimages, numofchannels, xdim, ydim] numpy array
    def extract_feature_from_multiple_images(self, imageArray):
        self.featureArray = []
        for i in imageArray:
            featureArray.append(self.extractor.extract_feature(i))
        return np.array(featureArray)
    # need to save previous featureArray for classification
    # and avg of all the euclidean dists of each feat to test feat for score
    def classify_single_image(image):
        testfeature = extract_feature_from_single_image(image)
        model = Naive_Euclid()
        model.fit(featureArray, self.trainY)
        predlabel, score = model.predict(testfeature)
        return predlabel, score

    def classify_multiple_images(imageArray):
        testfeatureArray = extract_feature_from_multiple_images(imageArray)
        model = Naive_Euclid()
        model.fit(testfeatureArray, self.trainY)
        predlabelArray = []
        scoreArray = []
        for feature in featureArray:
            templabel, tempscore = model.predict(feature)
            predlabelArray.append(templabel)
            scoreArray.append(tempscore)

        return predlabelArray, scoreArray
