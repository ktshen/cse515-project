from pathlib import Path
import sys
import os
import cv2 as cv
from module.database import FilesystemDatabase
from models import modelFactory
import argparse

parser = argparse.ArgumentParser(description="Phase 1 Task 2")
parser.add_argument(
    "-i",
    "--input_path",
    metavar="input_path",
    type=str,
    help="Input folder.",
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    metavar="model",
    type=str,
    help="The model will be used.",
    required=True,
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

inputPath = Path(args.input_path)
modelName = args.model.lower()
table = args.table.lower()

# Create filesystem database with SIFT
db = FilesystemDatabase(f"{table}_{modelName}", create=True)

model = modelFactory.creatModel(modelName)

# Remove unuse variable in case misusing.
del modelName
del table

SUPPORT_FILE_TYPES = [".jpg"]

allFiles = []

for fileName in os.listdir(inputPath):
    for extension in SUPPORT_FILE_TYPES:
        if fileName.endswith(extension):
            allFiles.append(inputPath / fileName)
            break

for i, imgFile in enumerate(allFiles):
    # Convert Path to str since imread do not recognize Path
    if not db.getData(imgFile.stem):
        img = cv.imread(str(imgFile))
        features = model.extractFeatures(img)
        featuresData = model.serializeFeature(features)
        db.addData(imgFile.stem, featuresData)
        print(f"{len(allFiles)}: {i+1} processed.")
