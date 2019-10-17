import cv2
import numpy as np
from .model import Model


class HOG(Model):
    def __init__(
        self, numOfBins=9, cellSize=8, isSign=False, l2HysThreshold=0.2, downSample=10
    ):
        self._numOfBins = numOfBins
        self._cellSize = cellSize
        self._blockSize = 2 * self._cellSize
        self._blockStride = self._blockSize // 2
        self._isSign = isSign
        self._l2HysThreshold = l2HysThreshold
        self._downSample = downSample

    def extractFeatures(self, img):
        height, width = img.shape[:2]
        # down sample image
        scaledImage = cv2.resize(
            img,
            (width // self._downSample, height // self._downSample),
            interpolation=cv2.INTER_AREA,
        )

        hog = cv2.HOGDescriptor(
            (width // self._downSample, height // self._downSample),
            (self._blockSize, self._blockSize),
            (self._blockStride, self._blockStride),
            (self._cellSize, self._cellSize),
            self._numOfBins,
        )

        vector = (
            hog.compute(scaledImage)
            .ravel()
            .reshape(
                (width // self._downSample - self._blockSize) // self._blockStride + 1,
                (height // self._downSample - self._blockSize) // self._blockStride + 1,
                self._numOfBins * (self._blockSize // self._cellSize) ** 2,
            )
        )

        return vector

    def getSimilarity(self, feature1, feature2, distanceFunction):
        pass

    def getSimilarityScore(self, data):
        pass

    def serializeFeature(self, featuresData):
        return featuresData

    def deserializeFeature(self, data):
        return data

    def visualizeFeatures(self, img, feature):
        print("Notice: the blocks are showed in the UP-DOWN then LEFT-RIGHT form")
        for i in range(0, len(feature)):
            for j in range(0, len(feature[i])):
                print("block " + str((i, j)))
                print(feature[i][j])

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
            flatFeatureList.append(self.flattenFecture(feature))

        featureMatrix = np.array(flatFeatureList)

        return dimRed(featureMatrix, k)

    def flattenFecture(self, feature, dimRedName = None):
        return feature.ravel()
