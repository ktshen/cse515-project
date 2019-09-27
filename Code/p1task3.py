import sys
from module.database import FilesystemDatabase
from module.distance import Norm
from pathlib import Path
import cv2 as cv
from models import modelFactory
import argparse

if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} FILEPATH MODEL TABLE_NAME K")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Phase 1 Task 2")
parser.add_argument(
    "-i",
    "--input_filepath",
    metavar="input_filepath",
    type=str,
    help="Input file path.",
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
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=True)
parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=True,
)

args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}")

# Create model
model = modelFactory.creatModel(model)

# The target image
imgPath = Path(args.input_filepath)
imgFolderPath = imgPath.parent
targetImageId = imgPath.stem
targetImg = cv.imread(str(imgPath))
k = int(args.topk)

targetFeature = model.deserializeFeature(db.getData(targetImageId))

# Distance function
l2Norm = Norm(2)

# Store (matching score, ID, kp, desc)
queryList = []

# Calculate the score of the target and each image
for keyId in db.keys():
    if keyId == targetImageId:
        continue

    queryFeature = model.deserializeFeature(db.getData(keyId))

    similarityData = model.getSimilarity(targetFeature, queryFeature, l2Norm)
    similarityScore = model.getSimilarityScore(similarityData)

    query = (similarityScore, keyId, queryFeature, similarityData)

    queryList.append(query)

# Sory by the score
queryList.sort(reverse=model.sortSimilarityScoreReverse())

# Output the top-k images
for i in range(k):
    similarityScore, queryKeyId, queryFeature, similarityData = queryList[i]

    queryImg = cv.imread(str(imgFolderPath / (queryKeyId + ".jpg")))

    visualizedImg = model.visualizeSimilarityResult(
        targetImg,
        targetFeature,
        queryImg,
        queryFeature,
        similarityData,
        i + 1,  # rank
        similarityScore
    )

    cv.imwrite(f"{i + 1}_{queryKeyId}_{similarityScore}.jpg", visualizedImg)
