import numpy as np
import sklearn.decomposition as sk
from abc import ABC, abstractmethod


class DimRed(ABC):

    methods = {}

    @abstractmethod
    def __init__(self, k, data):
        pass

    @abstractmethod
    def printLatentSemantics(self):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @classmethod
    def registerMethod(cls, method):
        DimRed.methods[method.__name__.lower()] = method

    @staticmethod
    def getSupportedMethods():
        return list(DimRed.methods.keys())

    @staticmethod
    def createReduction(method, **kwargs):
        if method.lower() in DimRed.methods:
            return DimRed.methods[method](**kwargs)
        else:
            raise Exception("Not supported dimension reduction method.")


@DimRed.registerMethod
class SVD(DimRed):
    def __init__(self, k, data):
        self.svd = sk.TruncatedSVD(n_components=k)
        self.svd.fit(data)

    def printLatentSemantics(self):
        print("The SVD latent semantics are: (weight, term)")
        for weight, term in zip(self.svd.explained_variance_, self.svd.components_):
            print((weight, term))

    def transform(self, data):
        return self.svd.transform(data)

@DimRed.registerMethod
class NMF(DimRed):
    def __init__(self, k, data):
        self.nmf = sk.NMF(n_components=k)
        self.nmf.fit(data)

    def printLatentSemantics(self):
        print("The NMF latent semantics are: (order, term)")
        order = 1
        for term in self.nmf.components_:
            print((order, term))
            order += 1

    def transform(self, data):
        return self.nmf.transform(data)


@DimRed.registerMethod
class PCA(DimRed):
    def __init__(self, k, data):
        self.pca = sk.PCA(n_components=k)
        self.pca.fit(data)

    def printLatentSemantics(self):
        print("The PCA latent semantics are: (weight, term)")
        for weight, term in zip(self.pca.explained_variance_, self.pca.components_):
            print((weight, term))

    def transform(self, data):
        return self.pca.transform(data)


@DimRed.registerMethod
class LDA(DimRed):
    def __init__(self, k, data):
        self.lda = sk.LatentDirichletAllocation(n_components=k)
        self.lda.fit(data)

    def printLatentSemantics(self):
        print("The LDA latent semantics are: (order, term)")
        order = 1
        for term in self.lda.components_:
            print((order, term))
            order += 1

    def transform(self, data):
        return self.lda.transform(data)
