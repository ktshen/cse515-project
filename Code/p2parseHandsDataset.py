import csv
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Phase 1 Task 1")
parser.add_argument(
    "-i",
    "--metadata_filepath",
    metavar="metadata_filepath",
    type=str,
    help="The filepath of metadata.",
    required=True,
)
parser.add_argument(
    "-D",
    "--dataset_folderpath",
    metavar="dataset_folderpath",
    type=str,
    help="The folder path of dataset which contains *.jpg files.",
    required=True,
)

metadata_labels = [
    ("l", "left_hand"),
    ("r", "right_hand"),
    ("d", "dorsal"),
    ("p", "palmar"),
    ("a", "accessories"),
    ("n", "no_accessories"),
    ("m", "male"),
    ("f", "female"),
]

labels_group = parser.add_argument_group("Label", "Choose the labels you want to filter. This is boolean OR operation. Which mean if you take male and female at same time, you will get all images.")

for label in metadata_labels:
    labels_group.add_argument(f"-{label[0]}", f"--{label[1]}", action="store_true")

args = parser.parse_args()

datasetPath = Path(args.dataset_folderpath)

with open(args.metadata_filepath, "r") as metadataFile:
    csvReader = csv.reader(metadataFile, delimiter=',')

    # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
    next(csvReader, None)  # skip the headers

    resultList = []

    for row in csvReader:
        if args.accessories and row[4] == '1':
            resultList.append(row)
            continue
        if args.no_accessories and row[4] == '0':
            resultList.append(row)
            continue
        if args.male and row[2] == 'male':
            resultList.append(row)
            continue
        if args.female and row[2] == 'female':
            resultList.append(row)
            continue
        
        aspectOfHand = row[6].strip().split(' ')

        if args.dorsal and aspectOfHand[0] == "dorsal":
            resultList.append(row)
            continue
        if args.palmar and aspectOfHand[0] == "palmar":
            resultList.append(row)
            continue
        if args.left_hand and aspectOfHand[1] == "left":
            resultList.append(row)
            continue
        if args.right_hand and aspectOfHand[1] == "right":
            resultList.append(row)
            continue

for hand in resultList:
    print(datasetPath / hand[7])

print(f"total files: {len(resultList)}")
