import csv
from collections import defaultdict


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


if __name__ == "__main__":
    print(getFilelistByID("/Users/bdfish/hw/cse515_data/HandInfo.csv")[0])
