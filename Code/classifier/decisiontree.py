from .classifier import Classifier


class DecisionTree(Classifier):
    def __init__(self, dimension=None, visualization=False):
        pass

    def fit(self, data, gt):
        # The data format should be:
        # data = [[features of data1], [features of data2], ... [features of dataN]]
        # gt = [y1, y2, y3, y4 ...] for each yi is true / false.
        pass

    def predict(self, data):
        pass
