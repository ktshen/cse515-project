from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data, gt):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @staticmethod
    def createClassifier(classifier, **kwargs):
        from .svm import SVM
        from .decisiontree import DecisionTree
        from .ppr import PPR
        classifiers = {"svm": SVM, "dtree": DecisionTree, "ppr": PPR}
        if classifier.lower() in classifiers:
            return classifiers[classifier](**kwargs)
        else:
            raise Exception("Not supported classifier.")
