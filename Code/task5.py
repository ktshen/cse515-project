from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
from module.handMetadataParser import getFilelistByLabel
import cv2
import numpy as np

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
parser.add_argument(
    "-d",
    "--method",
    metavar="method",
    type=str,
    help="The method will be used to reduce dimension.",
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
parser.add_argument(
    "-i",
    "--image_id",
    metavar="image_id",
    type=str,
    help="The unlabled image ID."
)
parser.add_argument(
    "-ip",
    "--image_path",
    metavar="image_path",
    type=str,
    help="The unlabled image ID."
)
args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk
decompMethod = args.method.lower()
distFunction = args.distance.lower()
metadataPath = args.metadata
label = args.label   # Used to filter such as left-hand or right-hand

# The first one is a image id in dataset. The second one is a path to load an image.
unlabeledImageID = args.image_id
unlabeledImagePath = args.image_path

opposite_label = {"l" : "r", "r" : "l",
                  "d" : "p", "p" : "d",
                  "a" : "n", "n" : "a",
                  "m" : "f", "f" : "m"
                  }
# Get filelist according to the label
labelImg = getFilelistByLabel(metadataPath, label)
oppoImg = getFilelistByLabel(metadataPath, opposite_label[label])

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}", create=False)
model = modelFactory.creatModel(model)
# decompFunction = DimRed.createReduction(decompMethod, k=topk)
distance = distanceFunction.createDistance(distFunction)

# The imageIdList and featureList should can be mapped to each other.
labelFeatList = []
oppoFeatList = []
# Removed unsed variable in case misusing.
del table

# Load features of images
for keyId in labelImg:
    feature = db.getData(keyId)
    # If we found that the "unlabled image id" belong to a label.
    # Then skip it and do not put it into feature list.
    if feature is not None:
        labelFeatList.append(model.flattenFecture(model.deserializeFeature(feature)))

for keyId in oppoImg:
    feature = db.getData(keyId)
    # If we found that the "unlabled image id" belong to a label.
    # Then skip it and do not put it into feature list.
    if feature is not None:
        oppoFeatList.append(model.flattenFecture(model.deserializeFeature(feature)))
# decompData = model.dimensionReduction(featuresList, decompFunction)
latenModel = DimRed.createReduction(decompMethod, k=topk, data=labelFeatList)
print("labeled data in the latent space:")
labelFeat = latenModel.transform(labelFeatList)
# labelFeat = abs(labelFeat)
# labelFeat = np.linalg.norm(labelFeat, axis = 1)

print("minimum:")
print(np.amin(labelFeat, axis = 0))
print("maximum:")
print(np.amax(labelFeat, axis = 0))
print("mean:")
print(np.mean(labelFeat, axis = 0))
print("std:")
print(np.std(labelFeat, axis = 0))
print("opposite data in the latent space:")
oppoFeat = latenModel.transform(oppoFeatList)
# oppoFeat = abs(oppoFeat)
# oppoFeat = np.linalg.norm(oppoFeat, axis = 1)

print("minimum:")
print(np.amin(oppoFeat, axis = 0))
print("maximum:")
print(np.amax(oppoFeat, axis = 0))
print("mean:")
print(np.mean(oppoFeat, axis = 0))
print("std:")
print(np.std(oppoFeat, axis = 0))