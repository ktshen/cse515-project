from abc import ABC, abstractmethod
import os
from pathlib import Path
import pickle
import glob


# An abstract class for storage
class Database(ABC):
    @abstractmethod
    def __init__(self, tableId):
        pass

    @abstractmethod
    def addData(self, key, data, overwrite=False):
        """[summary]

        Arguments:
            key (int or Path or str) -- Used as index key in database
            data -- any data which could be dumped by pickle

        Keyword Arguments:
            overwrite {bool} -- wheter we should overwrite the data if it was exists (default: {False})
        """
        pass

    @abstractmethod
    def getData(self, key):
        pass

    @abstractmethod
    def keys(self):
        pass


# To store all data as pickle file.
class FilesystemDatabase(Database):
    def __init__(self, tableId):
        super(Database, FilesystemDatabase).__init__(self)
        DEFAULT_DB_PATH = Path("FileDB")
        self._destPath = DEFAULT_DB_PATH / tableId

        if os.path.exists(self._destPath):
            if not os.path.isdir(self._destPath):
                raise Exception(f"{self._destPath} exists but not a directory!")
        else:
            self._destPath.mkdir(parents=True, exist_ok=True)

    def _getDataPathByKey(self, key):
        if type(key) == str:
            key = key.replace("/", "_")
            key = key.replace("\\", "_")
        elif isinstance(key, Path):
            # key = str(key).replace("/", "_")
            # key = key.replace("\\", "_")
            # This will remove all parent path and its extension.
            key = key.stem
        # Even though the key can be integer, we still convert it to a string first.
        return self._destPath / (str(key) + ".pkl")

    def addData(self, key, data, overwrite=False):
        dataPath = self._getDataPathByKey(key)

        if os.path.exists(dataPath):
            if not overwrite:
                return False

        with open(dataPath, "wb") as fileHndl:
            pickle.dump(data, fileHndl, pickle.HIGHEST_PROTOCOL)

        return True

    def getData(self, key):
        dataPath = self._getDataPathByKey(key)

        data = None

        if os.path.exists(dataPath):
            with open(dataPath, "rb") as fileHndl:
                data = pickle.load(fileHndl)

        return data

    def keys(self):
        # Not a good way to return key list
        pklList = glob.glob(str(self._destPath / "*.pkl"))
        keyList = []
        for pkl in pklList:
            keyList.append(Path(pkl).stem)

        return keyList


if __name__ == "__main__":
    # storage = FilesystemDatabase("TestDB")
    # a = {1:"abc", 2:"def"}

    # storage.addData(1, a)

    # x = storage.getData(1)

    # print(x)
    storage = FilesystemDatabase("hands_sub_sift")
    k = storage.keys()
    print(k)
