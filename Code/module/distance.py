from abc import ABC, abstractmethod
import numpy as np


class distance(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data1, data2):
        pass


class Norm(distance):
    def __init__(self, ord):
        self._ord = ord

    def __call__(self, data1, data2):
        # The data should be a one-dimension vector.
        """
        summation = 0

        for v in data:
            summation += abs(v) ** self._ord

        return summation ** (1 / self._ord)
        """
        return np.asscalar(
            np.power(np.sum(np.power(np.absolute(data1 - data2), self._ord)), 1 / self._ord)
        )


if __name__ == "__main__":
    l2Norm = Norm(2)

    x = [2, 3]

    print(l2Norm(x))
