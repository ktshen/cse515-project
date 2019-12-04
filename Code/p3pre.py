import argparse
from pathlib import Path
import os
from models import modelFactory
from module.database import FilesystemDatabase
import cv2 as cv


parser = argparse.ArgumentParser(description="Phase 3 Pre-compututation for task 1 and 2")

parser.add_argument(
    "-p",
    "--labeled_image_path",
    metavar="labeled_image_path",
    type=str,
    help="The folder path of labeled images.",
    required=True
)
parser.add_argument(
    "-unp",
    "--unlabeled_image_path",
    metavar="unlabeled_image_path",
    type=str,
    help="The folder path of unlabeled images.",
    required=True
)

parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=True,
)

args = parser.parse_args()
labeledPath = Path(args.labeled_image_path)
unlabeledPath = Path(args.unlabeled_image_path)
table = args.table.lower()

usedModels = ["cavg", "hog"] #task 1 uses these three models

allFile = []

for fileName in os.listdir(labeledPath):
    if fileName.endswith(".jpg"):
        allFile.append(labeledPath / fileName)
for fileName in os.listdir(unlabeledPath):
    if fileName.endswith(".jpg"):
        allFile.append(unlabeledPath / fileName)
        
print(f"{len(allFile)} images.")

for modelName in usedModels:
    print(f"Building a table {table} for model {modelName}")
    db = FilesystemDatabase(f"{table}_{modelName}", create=True)
    model = modelFactory.creatModel(modelName)

    for i, imgFile in enumerate(allFile):
        if db.getData(imgFile.stem) is None:
            img = cv.imread(str(imgFile))
            features = model.extractFeatures(img)
            featuresData = model.serializeFeature(features)
            db.addData(imgFile.stem, featuresData)
            print(f"{i+1} processed.")

