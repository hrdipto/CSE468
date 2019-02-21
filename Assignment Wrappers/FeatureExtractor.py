import ColorHue_Histogram as ch
import OrientedGradients_Histogram as ogh

class FeatureExtractor():
    def extract_feature():
        try:
            return self.get_message_impl ()
        except Exception as detail:
            print ("Error:", detail)
        return None
    def get_message_impl (self):
        raise Exception ("Not Implemented by Parent, call Child object instead")


class RawFlattenedPixel(FeatureExtractor):
    def extract_feature(image):
        rawflatPix = image.flatten()
        return rawflatPix

class ColourHistogramExtractor(FeatureExtractor):
    def extract_feature(image):
        col_hist = ColorHue_Histogram.getColorHistogram(image)
        return col_hist

class HOGExtractor(FeatureExtractor):
    def extract_feature(image):
        newimage = np.moveaxis(image,1,3)   # uses a default kernel size of (8,8)
        og_hist = OrientedGradients_Histogram.generate_Hog_Descriptor(newimage, (4,4))
        return og_hist

class SIFTBoVWExtractor(FeatureExtractor):
    def extract_feature(image):
        print('to be Implemented')
