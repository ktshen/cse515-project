import argparse
import numpy as np
import os
import cv2
from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
from module.handMetadataParser import getFileListByAspectOfHand
from classifier.classifier import Classifier
from pathlib import Path

parser = argparse.ArgumentParser(description="Phase 3 Task 4")
parser.add_argument(
    "-c",
    "--classifier",
    metavar="classifier",
    type=str,
    help="Classifier",
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    metavar="model",
    type=str,
    help="The model will be used.",
    required=False,
)
parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=False,
)
parser.add_argument(
    "-ut",
    "--unlabeled_table",
    metavar="unlabeled_table",
    type=str,
    help="The unlabeled table will be used to test.",
    required=False,
)
parser.add_argument(
    "-d",
    "--method",
    metavar="method",
    type=str,
    help="The method will be used to reduce dimension.",
    required=False,
)
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=False)
parser.add_argument(
    "-dis",
    "--distance",
    metavar="distance",
    type=str,
    help="Distance function.",
    default="l2",
    required=False
)
parser.add_argument(
    "-limg",
    "--labeled_image_path",
    metavar="labeled_image_path",
    type=str,
    help="The folder path of labeled images.",
    required=False,
)
parser.add_argument(
    "-uimg",
    "--unlabeled_image_path",
    metavar="unlabeled_image_path",
    type=str,
    help="The folder path of unlabeled images.",
    required=False,
)
parser.add_argument(
    "-rgb",
    "--color_image",
    help="Convert loaded image to grey or not.",
    default=False,
    action='store_true',
    required=False
)
parser.add_argument(
    "-meta",
    "--metadata",
    metavar="METADATA_PATH",
    type=str,
    help="Path of metadata.",
    required=True
)
parser.add_argument(
    "-tmeta",
    "--test_metadata",
    metavar="TEST_METADATA",
    type=str,
    help="The metadata path of unlabeled / test folder.",
    required=False
)
args = parser.parse_args()

# extract argument
classiferName = args.classifier.lower()
classifier = Classifier.createClassifier(classiferName)

modelName = args.model.lower() if args.model else None
table = args.table.lower() if args.table else None
unlabeledTable = args.unlabeled_table.lower() if args.unlabeled_table else None
decompMethod = args.method.lower() if args.method else None
distFunction = args.distance.lower()
metadataPath = args.metadata

testMetadataPath = args.test_metadata
topk = args.topk

labeledImgPath = args.labeled_image_path
unlabeledImgPath = args.unlabeled_image_path
useColorImage = args.color_image

# Data
trainingData = []
trainingGT = []
testingData = []
testingGT = []

# Get all image id with its corresponding label
dorsalFileIDList, palmarFileIDList, fileIDToLabelDict = getFileListByAspectOfHand(metadataPath)

# Error checking. Parser.error will end this task here.
if table is not None and modelName is None:
    parser.error("Please give -m as model name.")
elif table is None and modelName is not None:
    parser.error("Please give -t as table name.")
elif table is None and labeledImgPath is None:
    parser.error("Regarding the input images, please give a table name by -t or give the image folder path by -limg")
if unlabeledTable is not None and modelName is None:
    parser.error("Please give -m as model name.")
if decompMethod and topk is None:
    parser.error("Please give -k as k for dimension reduction function.")


# Used for load raw image.
# This function will return document-term matrix and a label list.
# If the filenameToLabel is None, this function will return label as empty list.
def loadImagesAsDocTerm(imgPath, useColor, fileIDToLabel=None):
    docTerm = []

    imgPath = Path(imgPath)
    label = []

    for fileName in os.listdir(imgPath):
        if fileIDToLabel is not None:
            if fileName[:-4] not in fileIDToLabel:
                continue
            else:
                label.append(fileIDToLabel[fileName[:-4]])

        cv2Flag = cv2.IMREAD_COLOR if useColor else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(str(imgPath / fileName), cv2Flag)
        img = img.flatten()
        docTerm.append(img)

    return docTerm, label

# Load training data and ground truth
print("Loading training image data...")

if table is not None and modelName is not None:
    # Open database
    db = FilesystemDatabase(f"{table}_{modelName}", create=False)
    model = modelFactory.creatModel(modelName)

    for fileID, isDorsal in fileIDToLabelDict.items():
        trainingData.append(model.flattenFecture(model.deserializeFeature(db.getData(fileID)), decompMethod))
        trainingGT.append(isDorsal)

    trainingData = np.array(trainingData)
else:
    trainingData, trainingGT = loadImagesAsDocTerm(labeledImgPath, useColorImage, fileIDToLabelDict)
    trainingData = np.array(trainingData)


if decompMethod is not None and topk is not None:
    print("Doing dimension reduction on training data...")
    # Create latent semantics
    latentModel = DimRed.createReduction(decompMethod, k=topk, data=trainingData)
    # Transform data
    trainingData = latentModel.transform(trainingData)


# Process testing data
print("Loading test data...")

# Load test meta data if exist. We can use this data to calculate prediction accuracy.
testFileIDToLabelDict = None
testFileIDList = []
if testMetadataPath is not None:
    _, _, testFileIDToLabelDict = getFileListByAspectOfHand(testMetadataPath)

# Load test image data
if unlabeledTable is not None and modelName is not None:
    # Open database. We do not need to create model here since the model should be created before load labeled data.
    db = FilesystemDatabase(f"{unlabeledTable}_{modelName}", create=False)

    for fileID in db.keys():
        if testFileIDToLabelDict is not None:
            if fileID in testFileIDToLabelDict:
                testFileIDList.append(fileID)
                testingData.append(model.flattenFecture(model.deserializeFeature(db.getData(fileID)), decompMethod))
                testingGT.append(testFileIDToLabelDict[fileID])
        else:
            testFileIDList.append(fileID)
            testingData.append(model.flattenFecture(model.deserializeFeature(db.getData(fileID)), decompMethod))

    testingData = np.array(testingData)
else:
    testingData, testingGT = loadImagesAsDocTerm(unlabeledImgPath, useColorImage, testFileIDToLabelDict)
    testingData = np.array(testingData)


if decompMethod is not None and topk is not None:
    print("Doing dimension reduction on testing data...")
    testingData = latentModel.transform(testingData)

# Training classifier
print("Training Classifer by training data...")
classifier.fit(trainingData, trainingGT)

# Predict the testing data
print("Predict testing data...")
testingResult = classifier.predict(testingData)

for i in range(len(testFileIDList)):
    print(f"{testFileIDList[i]}: {'dorsal' if testingResult[i] else 'palmar'}")

# Calculate accuracy if we have testing labels
correctNum = 0

if len(testingGT) > 0:
    for i in range(len(testingGT)):
        if testingGT[i] == testingResult[i]:
            correctNum += 1

    print(f"Accuracy: {(correctNum / len(testingResult)) * 100}%")
