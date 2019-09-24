import sys
from module.database import FilesystemDatabase
from module.distance import Norm
from pathlib import Path
import cv2 as cv
from models.sift import SIFT
from models.cm import ColorMoments
from models.lbp import LocalBP
from models.hog import HOG

if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} FILEPATH MODEL TABLE_NAME K")
    sys.exit(1)

# Create database according to model and table name
db = FilesystemDatabase(sys.argv[3].lower() + "_" + sys.argv[2].lower())

if sys.argv[2].lower() == "sift":
    # SIFT
    model = SIFT()
elif sys.argv[2].lower() == "cm":
    # Color Moments
    model = ColorMoments()
elif sys.argv[2].lower() == "lbp":
    # Local BP
    model = LocalBP()
elif sys.argv[2].lower() == "hog":
    # HOG
    model = HOG()
else:
    print("The model can only be CM or SIFT.")
    sys.exit(0)

# The target image
imgPath = Path(sys.argv[1])
imgFolderPath = imgPath.parent
targetImageId = imgPath.stem
targetImg = cv.imread(str(imgPath))
k = int(sys.argv[4])

targetFeature = model.deserializeFeature(db.getData(targetImageId))

# Distance function
l2Norm = Norm(2)

# Store (matching score, ID, kp, desc)
queryList = []

for keyId in db.keys():
    if keyId == targetImageId:
        continue

    queryFeature = model.deserializeFeature(db.getData(keyId))

    similarityData = model.getSimilarity(targetFeature, queryFeature, l2Norm)
    similarityScore = model.getSimilarityScore(similarityData)

    query = (similarityScore, keyId, queryFeature, similarityData)

    queryList.append(query)

queryList.sort(reverse=model.sortSimilarityScoreReverse())

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