from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extractFeatures(self, img):
        pass

    @abstractmethod
    def getSimilarity(self, feature1, feature2, distanceFunction):
        pass

    @abstractmethod
    def getSimilarityScore(self, data):
        pass

    @abstractmethod
    def serializeFeature(self, featuresData):
        pass

    @abstractmethod
    def deserializeFeature(self, data):
        pass

    @abstractmethod
    def visualizeFeatures(self, img, feature):
        pass

    @abstractmethod
    def visualizeSimilarityResult(
        self, img1, features1, img2, features2, similarityData, rank=0, score=0
    ):
        pass

    @abstractmethod
    def sortSimilarityScoreReverse(self):
        pass

    @abstractmethod
    def dimensionReduction(self, featureList, dimRed, k=None):
        pass

    @abstractmethod
    def flattenFecture(self, feature, dimRedName=None):
        pass
