from .model import Model
import cv2
import numpy as np

class ColorAvg(Model):
    def __init__(self, widthOfWindow=100, heightOfWindow=100):
        self._widthOfWindow = widthOfWindow
        self._heightOfWindow = heightOfWindow

    def extractFeatures(self, img):
        # Convert to YUV first.
        imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Get the height / width / number of channels
        h = imgYUV.shape[0]
        w = imgYUV.shape[1]
        c = imgYUV.shape[2]

        # The number of windows
        numOfRows = int((h + 1) / self._heightOfWindow)
        numOfColumns = int((w + 1) / self._widthOfWindow)

        # The results we wanted.
        resultMean = np.zeros(shape=(numOfRows, numOfColumns, c))
#         resultStd = np.zeros(shape=(numOfRows, numOfColumns, c))
#         resultSkew = np.zeros(shape=(numOfRows, numOfColumns, c))

        for hIdx in range(numOfRows):
            # Find the end index of row window.
            hEndIdx = (
                ((hIdx + 1) * self._heightOfWindow) if hIdx < numOfRows - 1 else None
            )

            for wIdx in range(numOfColumns):
                # Find the end index of column window.
                wEndIdx = (
                    ((wIdx + 1) * self._widthOfWindow)
                    if wIdx < numOfColumns - 1
                    else None
                )

                window = imgYUV[
                    hIdx * self._heightOfWindow : hEndIdx,
                    wIdx * self._widthOfWindow : wEndIdx,
                    :,
                ]

                resultMean[hIdx, wIdx, :] = np.mean(window, axis=(0, 1))
#                 resultStd[hIdx, wIdx, :] = np.std(window, axis=(0, 1))
#                 # I tried scipy.stats.skew, but it seems the result is not correct.
#                 # Implement it by myself.
#                 resultSkew[hIdx, wIdx, 0] = np.sum(
#                     np.power(window[:, :, 0] - resultMean[hIdx, wIdx, 0], 3)
#                 ) / (self._heightOfWindow * self._widthOfWindow)
#                 resultSkew[hIdx, wIdx, 1] = np.sum(
#                     np.power(window[:, :, 1] - resultMean[hIdx, wIdx, 1], 3)
#                 ) / (self._heightOfWindow * self._widthOfWindow)
#                 resultSkew[hIdx, wIdx, 2] = np.sum(
#                     np.power(window[:, :, 2] - resultMean[hIdx, wIdx, 2], 3)
#                 ) / (self._heightOfWindow * self._widthOfWindow)
# 
#                 resultSkew[hIdx, wIdx, 0] = np.sign(resultSkew[hIdx, wIdx, 0]) * (
#                     np.abs(resultSkew[hIdx, wIdx, 0])
#                 ) ** (1 / 3)
#                 resultSkew[hIdx, wIdx, 1] = np.sign(resultSkew[hIdx, wIdx, 1]) * (
#                     np.abs(resultSkew[hIdx, wIdx, 1])
#                 ) ** (1 / 3)
#                 resultSkew[hIdx, wIdx, 2] = np.sign(resultSkew[hIdx, wIdx, 2]) * (
#                     np.abs(resultSkew[hIdx, wIdx, 2])
#                 ) ** (1 / 3)

#         return (resultMean, resultStd, resultSkew)
        return resultMean


    def getSimilarity(self, feature1, feature2, distanceFunction):
        return None

    def getSimilarityScore(self, data):
        return data

    def serializeFeature(self, featuresData):
        return featuresData

    def deserializeFeature(self, data):
        return data

    def extendChannel(self, channel):
        return np.reshape(
            np.repeat(channel, 3), (channel.shape[0], channel.shape[1], 3)
        )

    def visualizeFeatures(self, img, feature):        
        return None

    def visualizeSimilarityResult(
        self, img1, features1, img2, features2, similarityData, rank=0, score=0
    ):        
        return None

    def sortSimilarityScoreReverse(self):
        return False

    def dimensionReduction(self, featureList, dimRed, k=None):
        flatFeatureList = []
        for feature in featureList:
            flatFeatures = self.flattenFecture(feature, type(dimRed).__name__)

            flatFeatureList.append(flatFeatures)

        featureMatrix = np.concatenate(flatFeatureList)

        return dimRed(featureMatrix, k)

    def flattenFecture(self, feature, dimRedName=None):
        colorAverage = np.reshape(feature, (1, -1))[0]
        return colorAverage
        
