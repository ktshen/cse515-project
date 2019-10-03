import csv


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


if __name__ == "__main__":
    import sys

    print(len(getFilelistByLabel(sys.argv[1], sys.argv[2])))
