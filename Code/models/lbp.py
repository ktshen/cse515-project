import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from .model import Model


class LocalBP(Model):

    # This Class is used to store features data of this model.
#     class LocalBPData:
#         def __init__(self, lbp, subLbp):
#             self._lbp = lbp
#             self._subLbp = subLbp
# 
#         def getLbp(self):
#             return self._lbp
# 
#         def getSubLbp(self):
#             return self._subLbp

    def __init__(
        self,
        widthOfWindow=100,
        heightOfWindow=100,
        numOfPoints=8,
        radius=2,
        method="uniform",
    ):
        # model parameters
        self._widthOfWindow = widthOfWindow
        self._heightOfWindow = heightOfWindow
        self._numOfPoints = numOfPoints
        self._radius = radius
        self._method = method

    def extractFeatures(self, img):
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(
            grayImg, self._numOfPoints, self._radius, self._method
        )
        lbpMaxValue = int(lbp.max())
        subLbp = []

        # split lbp image into 100 * 100 window then compute histograms
        for i in range(grayImg.shape[0] // self._heightOfWindow):
            for j in range(grayImg.shape[1] // self._widthOfWindow):
                subLbp.append(
                    np.histogram(
                        lbp[
                            i * self._heightOfWindow : (i + 1) * self._heightOfWindow,
                            j * self._widthOfWindow : (j + 1) * self._widthOfWindow,
                        ].ravel(),
                        bins=lbpMaxValue + 1,
                        range=(0, lbpMaxValue + 1),
                    )[0]
                )
        subLbp = np.array(subLbp).reshape(
            grayImg.shape[0] // self._heightOfWindow,
            grayImg.shape[1] // self._widthOfWindow,
            lbpMaxValue + 1,
        )

#         return LocalBP.LocalBPData(lbp, subLbp)
        return subLbp

    def getSimilarity(self, feature1, feature2, distanceFunction):
        pass

    def getSimilarityScore(self, data):
        pass

    def serializeFeature(self, featuresData):
#         return featuresData.getSubLbp()
        return featuresData

    def deserializeFeature(self, data):
        return data

    def visualizeFeatures(self, img, feature):
#         v = feature.getSubLbp()
        v = feature
        print("Notice: the sub areas are showed in the LEFT-RIGHT then UP-DOWN form")
        for i in range(0, len(v)):
            for j in range(0, len(v[i])):
                print("Sub-area " + str((i, j)))
                print(v[i][j])
                
#         return feature.getLbp()
        return img

    def visualizeSimilarityResult(
        self, img1, features1, img2, features2, similarityData, rank=0, score=0
    ):
        pass

    def sortSimilarityScoreReverse(self):
        pass

    def dimensionReduction(self, featureList, dimRed, k=None):
        flatFeatureList = []

        for feature in featureList:
            flatFeatureList.append(feature.ravel())

        featureMatrix = np.array(flatFeatureList)

        return dimRed(featureMatrix, k)
