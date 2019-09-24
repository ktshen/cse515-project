from pathlib import Path
import sys
import os
import cv2 as cv
from module.database import FilesystemDatabase
from models.sift import SIFT
from models.cm import ColorMoments
from models.lbp import LocalBP
from models.hog import HOG


if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} INPUT_DIR MODEL TABLE_NAME")
    sys.exit(1)

if sys.argv[2].lower() == "sift":
    # SIFT
    model = SIFT()
elif sys.argv[2].lower() == "cm":
    # Color Moments
    model = ColorMoments()
elif sys.argv[2].lower() == "lbp":
    # LocalBP
    model = LocalBP()
elif sys.argv[2].lower() == "hog":
    # HOG
    model = HOG()
else:
    print(f"Please assign model: sift or cm")
    sys.exit(1)

# Create filesystem database with SIFT
db = FilesystemDatabase(f"{sys.argv[3]}_{sys.argv[2].lower()}")

SUPPORT_FILE_TYPES = [".jpg"]

inputPath = Path(sys.argv[1])
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
