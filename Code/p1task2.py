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
table = args.table.lower()

# All models is a list stored all model we would like to use this time.
# If the model argument are None, the build all model.
if args.model:
    allModels = [args.model.lower()]
else:
    allModels = modelFactory.getSupportModel()

for modelName in allModels:
    print(f"Building a table {table} for model {modelName}")
    # Create filesystem database with SIFT
    db = FilesystemDatabase(f"{table}_{modelName}", create=True)

    model = modelFactory.creatModel(modelName)

    SUPPORT_FILE_TYPES = [".jpg"]

    allFiles = []

    for fileName in os.listdir(inputPath):
        for extension in SUPPORT_FILE_TYPES:
            if fileName.endswith(extension):
                allFiles.append(inputPath / fileName)
                break

    for i, imgFile in enumerate(allFiles):
        # Convert Path to str since imread do not recognize Path
        if db.getData(imgFile.stem) is None:
            img = cv.imread(str(imgFile))
            features = model.extractFeatures(img)
            featuresData = model.serializeFeature(features)
            db.addData(imgFile.stem, featuresData)
            print(f"{len(allFiles)}: {i+1} processed.")
