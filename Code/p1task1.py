import cv2
import sys
from module.database import FilesystemDatabase
from pathlib import Path
from models import modelFactory
import argparse

parser = argparse.ArgumentParser(description="Phase 1 Task 1")
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
args = parser.parse_args()

# create model
model = modelFactory.creatModel(args.model.lower())

# Load image
filenamePath = Path(args.input_filepath)
key = filenamePath.stem
img = cv2.imread(str(filenamePath))

# Load features
features = model.extractFeatures(img)

# Show visualized features on image
img = model.visualizeFeatures(img, features)

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
cv2.imshow("Features", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.jpg", img)
