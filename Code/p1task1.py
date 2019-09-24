import cv2
import sys
from module.database import FilesystemDatabase
from pathlib import Path
from models.sift import SIFT
from models.cm import ColorMoments
from models.lbp import LocalBP
from models.hog import HOG


if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} FILEPATH MODEL")
    sys.exit(0)

if sys.argv[2].lower() == "cm":
    # Color Moments model
    model = ColorMoments()
elif sys.argv[2].lower() == "sift":
    # SIFT model
    model = SIFT()
elif sys.argv[2].lower() == "lbp":
    # LocalBP model
    model = LocalBP()
elif sys.argv[2].lower() == "hog":
    # HOG
    model = HOG()
else:
    print("The model can only be CM or SIFT.")
    sys.exit(0)

# Load image
filenamePath = Path(sys.argv[1])
key = filenamePath.stem
img = cv2.imread(str(filenamePath))

# Load features
features = model.extractFeatures(img)

# Show visualized features on image
img = model.visualizeFeatures(img, features)

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
cv2.imshow('Features', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output.jpg', img)
