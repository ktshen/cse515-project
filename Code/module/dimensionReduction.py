import numpy as np
from sklearn.decomposition import PCA as SKPCA
from sklearn.decomposition import LatentDirichletAllocation as SKLDA
from abc import ABC, abstractmethod


class DimReduction(ABC):
    @abstractmethod
    def __init__(self, k):
        pass

    @abstractmethod
    def __call__(self, data1, data2):
        pass

    @abstractmethod
    def getTermWeight(self, data, topk):
        pass

    @abstractmethod
    def projectFeature(self, feature, data, topk):
        pass

    @abstractmethod
    def getObjLaten(self, data, topk):
        pass

    @staticmethod
    def createReduction(method, **kwargs):
        # Add new method here
        methods = {"svd": SVD, "pca": PCA, "lda": LDA}

        if method.lower() in methods:
            return methods[method](**kwargs)
        else:
            raise Exception("Not supported dimension reduction method.")


class SVD(DimReduction):
    def __init__(self, k=None):
        self._topK = k

    def __call__(self, data, k=None):
        # data is a matrix. Each row represent feature vector.
        k = self._topK if k is None else k
        U, s, V = np.linalg.svd(data, full_matrices=False)

        return (U, s, V)

    def getTermWeight(self, data, topk):
        _, s, _ = data
        return np.diag(s[: topk])

    def projectFeature(self, feature, data, topk):
        pass

    def getObjLaten(self, data, topk):
        return data[0][:, : topk]


class PCA(DimReduction):
    def __init__(self, k=None):
        self._topK = k

    def __call__(self, data, k=None):
        # data is a matrix. Each row represent feature vector.
        k = self._topK if k is None else k
        pca = SKPCA().fit(data)
        dataTransform = pca.transform(data)
        s = pca.explained_variance_
        V = pca.components_
        U = V.T
        C = pca.get_covariance()

        return (dataTransform, s, V)

    def getTermWeight(self, data, topk):
    #task 1,3
        _, s, _ = data
        return np.diag(s[:topk])
        pass

    def projectFeature(self, feature, data, topk):
    #task 5
        pass

    def getObjLaten(self, data, topk):
    #task 2,4
        return data[0][:, :topk]


class LDA(DimReduction):
    def __init__(self, k=None):
        self._topK = k

    def __call__(self, data, k=None):
        # data is a matrix. Each row represent feature vector.
        k = self._topK if k is None else k
        lda = SKLDA(n_components=k, n_jobs=-1).fit(data)
        picTop = lda.transform(data)
        topFeature = lda.components_
        # weight = lda.exp_dirichlet_component_

        return [picTop, topFeature, k, data]

    def getTermWeight(self, data, topk):
        if data[2] != topk:
            picTop, topFeature, topk, feature = self.__call__(data[3], topk)
            data[0] = picTop
            data[1] = topFeature
            data[2] = topk
        return data[1]

    def projectFeature(self, feature , data, topk):
        pass

    def getObjLaten(self, data, topk):
        if data[2] != topk:
            picTop, topFeature, topk, feature = self.__call__(data[3], topk)
            data[0] = picTop
            data[1] = topFeature
            data[2] = topk
        return data[0]
