import numpy as np
from tqdm import tqdm
from . import classifier
import pickle


class SVM(classifier.Classifier):
    def __init__(self, dimension=None, visualization=False, pretrained=None, canFineTune=False):
        self._visualization = visualization
        self._color = {1: "r", -1: "b"}

        self._eta = 0.2
        self._regularization = 0.1
        self._epoch = 100
        self._canFineTune = True

        if pretrained is not None:
            with open(pretrained, "rb") as fhndl:
                self._w = pickle.load(fhndl)

            # only consider when we have pretrained weight.
            self._canFineTune = canFineTune
        elif dimension is not None:
            # 1 is B
            self._w = np.random.random([dimension + 1]) * 2 - 1.0
        else:
            self._w = None

    def fit(self, data, gt):
        if not self._canFineTune:
            return

        # The data format should be:
        # data = [[features of data1], [features of data2], ... [features of dataN]]
        # gt = [y1, y2, y3, y4 ...] for each yi is true / false.

        numOfData = data.shape[0]

        features = np.concatenate((data[:, :-1], np.ones((numOfData, 1))), axis=1)

        # Convert [true, false, ...] to [1.0, -1.0, ...]
        gt = np.array(gt) * 2 - 1.0

        # Initialize the weight to [-1.0, 1.0]
        self._w = 2.0 * np.random.random(features.shape[1]) - 1.0

        # The following [0, 1, -2] is an example in Ch12 of Mining of Massive Datasets.
        # self._w = np.array([0, 1, -2], dtype=np.float)

        for i in tqdm(range(self._epoch)):
            # This would get one dimension vector whose elements indicate the forward results.
            result = np.dot(self._w, features.T)

            # If the forward result is different to y, then it would be 1 in the following vecotr.
            # For example, we have 10 data and the forwared result of 2nd and 4th 4 are different to y:
            # [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
            result = 1.0 - (np.multiply(result, gt) >= 1.0)

            # The following code are gradient descent.
            result = np.multiply(features, result[:, np.newaxis])
            result = np.multiply(result, (gt * -1.0)[:, np.newaxis])
            self._w = self._w - self._eta * (self._w + self._regularization * np.sum(result, axis=0))

    def predict(self, data):
        if self._w is None:
            raise Exception("Please train SVM model first")

        return np.dot(self._w, data.T) > 0

    def save(self, filename):
        import pickle
        with open(f"svm/svm_{filename}.pkl", "wb") as fhndl:
            pickle.dump(self._w, fhndl)


if __name__ == "__main__":
    svm = SVM()
    data = 2 * np.random.random((10, 10)) - 1
    # The following [0, 1, -2] is an example in Ch12 of Mining of Massive Datasets.
    """
    data = np.array(
        [
            [1, 4, 1],
            [2, 2, 1],
            [3, 4, 1],
            [1, 1, -1],
            [2, 1, -1],
            [3, 1, -1],
        ]
    )
    """
    data[:, -1:] = data[:, -1:] / abs(data[:, -1:])
    svm.fit(data, None)
