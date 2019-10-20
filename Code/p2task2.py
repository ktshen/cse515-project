import sys
from module.database import FilesystemDatabase
from module.DimRed import DimRed
from pathlib import Path
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
import shutil

parser = argparse.ArgumentParser(description="Phase 2 Task 2")
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
args = parser.parse_args()

# extract argument
modelName = args.model.lower()
table = args.table.lower()
topk = args.topk
decompMethod = args.method.lower()
target = args.image_id
distFunction = args.distance.lower()
topm = args.topm
imagePath = Path(args.image_path)

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{modelName}", create=False)

# Create model and distance function
model = modelFactory.creatModel(modelName)
distance = distanceFunction.createDistance(distFunction)

# The imageIdList and featureList should can be mapped to each other.
featuresList = []
imageIdList = []
targetIdx = -1

# Removed unsed variable in case misusing.
del table

objFeat = []

# Load data and map queried index
for idx, keyId in enumerate(db.keys()):
    if target == keyId:
        targetIdx = idx
    imageIdList.append(keyId)
    objFeat.append(
        model.flattenFecture(model.deserializeFeature(db.getData(keyId)), decompMethod)
    )

# Exit if we cannot find the target image.
if targetIdx == -1:
    print(f"{target} is not in database.")
    sys.exit(1)

# Create latent semantics
latentModel = DimRed.createReduction(decompMethod, k=topk, data=objFeat)
# Transform data
resultMatrix = latentModel.transform(objFeat)
# Map queried feature
targetFeature = resultMatrix[targetIdx]

# This list will store (score, image ID)
distanceScoreList = []

for featureIdx in range(0, resultMatrix.shape[0]):
    # target image, skip
    if featureIdx == targetIdx:
        continue

    distanceScore = distance(targetFeature, resultMatrix[featureIdx])

    # Put score as first element of tuple so that we can make the score as key to sort this list.
    distanceScoreList.append((distanceScore, imageIdList[featureIdx]))

distanceScoreList.sort()

# Output results
outputFolder = Path(f"{modelName}_{decompMethod}_{topk}_{target}_{topm}")
outputFolder.mkdir(exist_ok=True)
for i in range(min(topm, len(distanceScoreList))):
    score = distanceScoreList[i][0]
    imageId = distanceScoreList[i][1]
    shutil.copyfile(
        imagePath / (imageId + ".jpg"), outputFolder / f"{i+1}_{imageId}_{score}.jpg"
    )
    print(f"Rank: {i+1}  ID: {imageId}  Score: {score}")

print(f"The result images have been written to folder {outputFolder}/.")
