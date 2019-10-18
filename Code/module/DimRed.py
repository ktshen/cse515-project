import numpy as np
import sklearn.decomposition as sk
from abc import ABC, abstractmethod
from _operator import index
import shutil
from pathlib import Path


class DimRed(ABC):

    methods = {}

    @abstractmethod
    def __init__(self, k, data):
        pass

    @abstractmethod
    def printLatentSemantics(self, ids, data, imagePath):
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

    def printLatentSemantics(self, ids, data, imagePath):
        outputFolder = Path("SVD latent semantics")
        outputFolder.mkdir(exist_ok=True)

        print("The SVD latent semantics are:(order, id, dot product value)")
        for order, ls in enumerate(self.svd.components_):
            maxIndex = 0
            maxProjection = 0
            for index, obj in enumerate(data):
                projection = np.dot(ls, obj)
                if projection > maxProjection:
                    maxIndex = index
                    maxProjection = projection
            print((order + 1, ids[maxIndex]), maxProjection)
            shutil.copyfile(
                imagePath / (ids[maxIndex] + ".jpg"), outputFolder / f"{order+1}latent semantics_{ids[maxIndex]}.jpg"
            )
        print(f"The result images have been written to folder {outputFolder}/.")


    def transform(self, data):
        return self.svd.transform(data)

@DimRed.registerMethod
class NMF(DimRed):
    def __init__(self, k, data):
        self.nmf = sk.NMF(n_components=k)
        self.nmf.fit(data)

    def printLatentSemantics(self, ids, data, imagePath=""):
        outputFolder = Path("NMF latent semantics")
        outputFolder.mkdir(exist_ok=True)
        print("The NMF latent semantics are:(order, id, dot product value)")
        for order, ls in enumerate(self.nmf.components_):
            maxIndex = 0
            maxProjection = 0
            for index, obj in enumerate(data):
                projection = np.dot(ls, obj)
                if projection > maxProjection:
                    maxIndex = index
                    maxProjection = projection
            print((order + 1, ids[maxIndex], maxProjection))
            if imagePath:
                shutil.copyfile(
                    imagePath / (ids[maxIndex] + ".jpg"), outputFolder / f"{order+1}latent semantics_{ids[maxIndex]}.jpg"
                )
                print(f"The result images have been written to folder {outputFolder}/.")

    def transform(self, data):
        return self.nmf.transform(data)


@DimRed.registerMethod
class PCA(DimRed):
    def __init__(self, k, data):
        self.pca = sk.PCA(n_components=k)
        self.pca.fit(data)

    def printLatentSemantics(self, ids, data, imagePath):
        outputFolder = Path("PCA latent semantics")
        outputFolder.mkdir(exist_ok=True)

        print("The PCA latent semantics are:(order, id, dot product value)")
        for order, ls in enumerate(self.pca.components_):
            maxIndex = 0
            maxProjection = 0
            for index, obj in enumerate(data):
                projection = np.dot(ls, obj)
                if projection > maxProjection:
                    maxIndex = index
                    maxProjection = projection
            print((order + 1, ids[maxIndex]), maxProjection)
            shutil.copyfile(
                imagePath / (ids[maxIndex] + ".jpg"), outputFolder / f"{order+1}latent semantics_{ids[maxIndex]}.jpg"
            )
        print(f"The result images have been written to folder {outputFolder}/.")

    def transform(self, data):
        return self.pca.transform(data)


@DimRed.registerMethod
class LDA(DimRed):
    def __init__(self, k, data):
        self.lda = sk.LatentDirichletAllocation(n_components=k)
        self.lda.fit(data)

    def printLatentSemantics(self, ids, data, imagePath):
        outputFolder = Path("LDA latent semantics")
        outputFolder.mkdir(exist_ok=True)

        print("The LDA latent semantics are:(order, id, dot product value)")
        for order, ls in enumerate(self.lda.components_):
            maxIndex = 0
            maxProjection = 0
            for index, obj in enumerate(data):
                projection = np.dot(ls, obj)
                if projection > maxProjection:
                    maxIndex = index
                    maxProjection = projection
            print((order + 1, ids[maxIndex]), maxProjection)
            shutil.copyfile(
                imagePath / (ids[maxIndex] + ".jpg"), outputFolder / f"{order+1}latent semantics_{ids[maxIndex]}.jpg"
            )
        print(f"The result images have been written to folder {outputFolder}/.")

    def transform(self, data):
        return self.lda.transform(data)
