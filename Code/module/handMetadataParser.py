import csv
from collections import defaultdict
import numpy as np


""" Get the file list by aspect of hand(dorsal/palmar)
    This function is used for phase 3.

Raises:
    Exception: [description]

Returns:
    [type] -- [description]
"""
def getFileListByAspectOfHand(metadataPath):

    dorsalFileIDList = []
    palmarFileIDList = []
    fileIDToLabelDict = {}

    with open(metadataPath, "r") as metadataFile:
        csvReader = csv.reader(metadataFile, delimiter=',')

        # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
        next(csvReader, None)  # skip the headers

        for row in csvReader:
            dorsal = row[7].strip().split(' ')[0] == "dorsal"

            fileID = row[8][:-4]

            if dorsal:
                dorsalFileIDList.append(fileID)
                fileIDToLabelDict[fileID] = True
            else:
                palmarFileIDList.append(fileID)
                fileIDToLabelDict[fileID] = False

    return dorsalFileIDList, palmarFileIDList, fileIDToLabelDict


def getFilelistByLabel(metadataPath, label):
    metadataLabelsChecker = {
        "l": lambda row: row[6].strip().split(' ')[1] == "left",
        "r": lambda row: row[6].strip().split(' ')[1] == "right",
        "d": lambda row: row[6].strip().split(' ')[0] == "dorsal",
        "p": lambda row: row[6].strip().split(' ')[0] == "palmar",
        "a": lambda row: row[4] == '1',
        "n": lambda row: row[4] == '0',
        "m": lambda row: row[2] == 'male',
        "f": lambda row: row[2] == 'female',
    }

    for c in label:
        if c not in metadataLabelsChecker:
            raise Exception(f"{label} is not a valid label.")

    filteredFileIDlist = []

    with open(metadataPath, "r") as metadataFile:
        csvReader = csv.reader(metadataFile, delimiter=',')

        # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
        next(csvReader, None)  # skip the headers

        for row in csvReader:
            for c in label:
                if metadataLabelsChecker[c](row):
                    filteredFileIDlist.append(row[7][:-4])
                    continue  # To ensure we only append one time.

    return filteredFileIDlist


def getFilelistByID(metadataPath):
    # return two dictionary:
    # a dictionary contains {subject ID, list:[image ID]}
    # a dictionary contains {image ID, subject ID}
    subToImg = defaultdict(list)
    imgToSub = {}

    with open(metadataPath, "r") as metadataFile:
        csvReader = csv.reader(metadataFile, delimiter=',')

        next(csvReader, None)  # skip the headers

        for row in csvReader:
            subjectId = int(row[0])
            imageId = row[7][:-4]
            subToImg[subjectId].append(imageId)
            imgToSub[imageId] = subjectId

    return subToImg, imgToSub


def getFilelistWithOneHot(metadataPath):
    # return a dictionary: image ID -> one-hot array
    imgToOnehot = {}

    with open(metadataPath, "r") as metadataFile:
        csvReader = csv.reader(metadataFile, delimiter=',')

        next(csvReader, None)  # skip the headers

        for row in csvReader:
            imageId = row[7][:-4]
            onehot = np.zeros(8)

            if row[6].strip().split(' ')[1] == "left":
                onehot[0] = 1.0
            else:
                onehot[1] = 1.0

            if row[6].strip().split(' ')[0] == "dorsal":
                onehot[2] = 1.0
            else:
                onehot[3] = 1.0

            if row[4] == '1':
                onehot[4] = 1.0
            else:
                onehot[5] = 1.0

            if row[2] == 'male':
                onehot[6] = 1.0
            else:
                onehot[7] = 1.0

            imgToOnehot[imageId] = onehot

    return imgToOnehot


if __name__ == "__main__":
    print(getFilelistByID("/Users/bdfish/hw/cse515_data/HandInfo.csv")[0])
