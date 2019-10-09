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
method = args.method.lower()
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
method = DimReduction.createReduction(method, k=topk)

# The imageIdList and featureList should could be mapped to each other.
featuresList = []
imageIdList = []

# Get target image feature
targetFeature = model.deserializeFeature(db.getData(target))

# Make the feature of target image at first in the list.
featuresList.append(targetFeature)
imageIdList.append(target)

if targetFeature is None:
    print(f"Cannot find {target} in table {table}")
    sys.exit(1)

# Removed unsed variable in case misusing.
del table

# Load features of images
for keyId in filteredFilelist:
    if target == keyId:
        continue

    feature = db.getData(keyId)

    # Because the database may store subset dataset only.
    # Some keyId may not exsit in database and therefore the feature would be None.
    if feature is not None:
        featuresList.append(model.deserializeFeature(feature))
        imageIdList.append(keyId)

dataTransform, _ = model.dimensionReduction(featuresList, method)

# TODO: Use U only or UsV?
# resultMatrix = np.dot(U, np.dot(s, V))
resultMatrix = dataTransform

# We always put target feature in first row.
targetFeature = resultMatrix[0]

distance = distanceFunction.createDistance(distFunction)

# Removed unused variable in case misusing
del featuresList

# This list will store (score, image ID)
distanceScoreList = []

# Because the first one(index: 0) is target, the range would be [1: length).
for featureIdx in range(1, resultMatrix.shape[0]):
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