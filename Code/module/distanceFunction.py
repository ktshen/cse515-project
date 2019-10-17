from abc import ABC, abstractmethod
import numpy as np
import scipy.stats.entropy as kvd


class distanceFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data1, data2):
        pass

    @staticmethod
    def createDistance(method, **kwargs):
        # Add new method here
        methods = {"l2", "l1", "kvd"}

        if method.lower() in methods:
            if method.lower() == "l2":
                return Norm(2)
            elif method.lower() == "l1":
                return Norm(1)
            elif method.lower() == "kvd":
                return Kvd()
        else:
            raise Exception("Not supported distance function method.")


class Norm(distanceFunction):
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

class Kvd(distanceFunction):
    def __init__(self):
        pass
    
    def __call__(self, data1, data2):
        
        return kvd(data1, data2)
        
if __name__ == "__main__":
    l2Norm = Norm(2)

    x = [2, 3]

    print(l2Norm(x))
