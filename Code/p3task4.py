import argparse
import numpy as np
from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
from module.handMetadataParser import getFileListByAspectOfHand
from classifier.classifier import Classifier

parser = argparse.ArgumentParser(description="Phase 3 Task 4")
parser.add_argument(
    "-c",
    "--classifier",
    metavar="classifier",
    type=str,
    help="Classifier",
    required=False,
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
    "-l",
    "--labeled_image_path",
    metavar="labeled_image_path",
    type=str,
    help="The folder path of labeled images.",
    required=False,
)
parser.add_argument(
    "-u",
    "--unlabeled_image_path",
    metavar="unlabeled_image_path",
    type=str,
    help="The folder path of unlabeled images.",
    required=False,
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

# Data
trainingData = []
trainingGT = []
testingData = []
testingGT = []

# Get all image id with its corresponding label
dorsalFileIDList, palmarFileIDList, fileIDToLabelDict = getFileListByAspectOfHand(metadataPath)

# Error checking.
if table is not None and modelName is None:
    # This task will end here.
    parser.error("Please give -m as model name.")
elif table is None and modelName is not None:
    # This task will end here.
    parser.error("Please give -t as table name.")
if unlabeledTable is not None and modelName is None:
    # This task will end here.
    parser.error("Please give -m as model name.")


if decompMethod and topk is None:
    # This task will end here.
    parser.error("Please give -k as k for dimension reduction function.")


# Load training data and ground truth
if table is not None and modelName is not None:
    # Open database
    db = FilesystemDatabase(f"{table}_{modelName}", create=False)
    model = modelFactory.creatModel(modelName)

    for fileID, isDorsal in fileIDToLabelDict.items():
        trainingData.append(model.flattenFecture(model.deserializeFeature(db.getData(fileID)), decompMethod))
        trainingGT.append(isDorsal)

    trainingData = np.array(trainingData)
else:
    pass   # Load image file directly


if decompMethod is not None and topk is not None:
    # Create latent semantics
    latentModel = DimRed.createReduction(decompMethod, k=topk, data=trainingData)
    # Transform data
    trainingData = latentModel.transform(trainingData)


# Process testing data

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
    pass   # Load image file directly


if decompMethod is not None and topk is not None:
    testingData = latentModel.transform(testingData)

# Training classifier
classifier.fit(trainingData, trainingGT)

# Predict the testing data
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
