import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
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
        # sekf._hog = cv2.HOGDescriptor()

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
            hog.comput(scaledImage)
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
        pass

    def deserializeFeature(self, data):
        pass

    def visualizeFeatures(self, img, feature):
        pass

    def visualizeSimilarityResult(
        self, img1, features1, img2, features2, similarityData, rank=0, score=0
    ):
        pass

    def sortSimilarityScoreReverse(self):
        pass
