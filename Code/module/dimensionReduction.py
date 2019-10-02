import numpy as np
from sklearn.decomposition import PCA as SKPCA
from abc import ABC, abstractmethod


class DimReduction(ABC):
    @abstractmethod
    def __init__(self, k):
        pass

    @abstractmethod
    def __call__(self, data1, data2):
        pass

    @staticmethod
    def createReduction(method, **kwargs):
        # Add new method here
        methods = {"svd": SVD, "pca": PCA}

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

        if k is not None:
            U, s, V = U[:, :k], np.diag(s[:k]), V[:k, :]

        return (U, s)


class PCA(DimReduction):
    def __init__(self, k=None):
        self._topK = k

    def __call__(self, data, k=None):
        # data is a matrix. Each row represent feature vector.
        k = self._topK if k is None else k
        pca = SKPCA(n_components=k).fit(data)
        dataTransform = pca.transform(data)
        s = np.diag(pca.explained_variance_)
        V = pca.components_
        U = V.T
        C = pca.get_covariance()

        return (dataTransform, s)
