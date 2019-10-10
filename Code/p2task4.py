import sys
from module.database import FilesystemDatabase
from module.dimensionReduction import DimReduction
from pathlib import Path
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
import shutil
from module.handMetadataParser import getFilelistByLabel


parser = argparse.ArgumentParser(description="Phase 2 Task 4")
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
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=True)
parser.add_argument("-mi", "--topm", metavar="topm", type=int, help="M.", required=True)
parser.add_argument(
    "-d",
    "--method",
    metavar="method",
    type=str,
    help="The method will be used to reduce dimension.",
    required=True,
)
parser.add_argument(
    "-p",
    "--image_path",
    metavar="image_path",
    type=str,
    help="The folder path of images.",
    required=True,
)
parser.add_argument(
    "-i",
    "--image_id",
    metavar="image_id",
    type=str,
    help="The target image ID.",
    required=True,
)
parser.add_argument(
    "-dis",
    "--distance",
    metavar="distance",
    type=str,
    help="Distance function.",
    default="l2",
)
parser.add_argument(
    "-l",
    "--label",
    metavar="label",
    type=str,
    help="""Label used for filter file list.\n
    l for left-hand,\n
    r for right-hand,\n
    d for dorsal,\n
    p for palmar,\n
    a for with accessories,\n
    n for without accessories,\n
    m for male,\n
    f for female.""",
    required=True
)
parser.add_argument(
    "-meta",
    "--metadata",
    metavar="METADATA_PATH",
    type=str,
    help="Path of metadata.",
    required=True
)
args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk
decompMethod = args.method.lower()
target = args.image_id
distFunction = args.distance.lower()
topm = args.topm
imagePath = Path(args.image_path)
metadataPath = args.metadata
label = args.label   # Used to filter such as left-hand or right-hand

# Get filelist according to the label
filteredFilelist = getFilelistByLabel(metadataPath, label)

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}", create=False)
model = modelFactory.creatModel(model)
decompFunction = DimReduction.createReduction(decompMethod, k=topk)
distance = distanceFunction.createDistance(distFunction)

# The imageIdList and featureList should can be mapped to each other.
featuresList = []
imageIdList = []
targetIdx = -1

# Removed unsed variable in case misusing.
del table

# Load features of images
for keyId in filteredFilelist:
    feature = db.getData(keyId)

    # Because the database may store subset dataset only.
    # Some keyId may not exsit in database and therefore the feature would be None.
    if feature is not None:
        if keyId == target:
            targetIdx = len(imageIdList)
        featuresList.append(model.deserializeFeature(feature))
        imageIdList.append(keyId)

decompData = model.dimensionReduction(featuresList, decompFunction)
resultMatrix = decompFunction.getObjLaten(decompData, topk)

targetFeature = resultMatrix[targetIdx]

# This list will store (score, image ID)
distanceScoreList = []

# Because the first one(index: 0) is target, the range would be [1: length).
for featureIdx in range(resultMatrix.shape[0]):
    # target image, skip
    if featureIdx == targetIdx:
        continue

    distanceScore = distance(targetFeature, resultMatrix[featureIdx])

    # Put score as first element of tuple so that we can make the score as key to sort this list.
    distanceScoreList.append((distanceScore, imageIdList[featureIdx]))

distanceScoreList.sort()

# TODO: We may need to find a new way to represent output.
outputFolder = Path("output")
outputFolder.mkdir(exist_ok=True)

for i in range(min(topm, len(distanceScoreList))):
    score = distanceScoreList[i][0]
    imageId = distanceScoreList[i][1]
    shutil.copyfile(
        imagePath / (imageId + ".jpg"), outputFolder / f"{i+1}_{imageId}_{score}.jpg"
    )
    print(f"Rank: {i+1}  ID: {imageId}  Score: {score}")

print(f"The result images have been written to folder {outputFolder}/.")
