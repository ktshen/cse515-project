from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
import cv2
import itertools


parser = argparse.ArgumentParser(description="Phase 2 Task 4")
parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=True,
)

parser.add_argument(
    "-i",
    "--image_id",
    metavar="image_id",
    type=str,
    help="The subject image ID.",
    required=True
)

args = parser.parse_args()

# extract argument
table = args.table.lower()
subjectImageID = args.image_id


# Get all models and dimention reduction methods
models = modelFactory.getSupportModel()
decompMethods = DimRed.getSupportedMethods()

for modelName, decompMethodName in itertools.product(models, decompMethods):
    print(f"modelName: {modelName} decompMethodName:{decompMethodName}")

import sys
sys.exit(0)

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}", create=False)
model = modelFactory.creatModel(model)
decompFunction = DimReduction.createReduction(decompMethod, k=topk)
distance = distanceFunction.createDistance(distFunction)

# The imageIdList and featureList should can be mapped to each other.
featuresList = []

# Removed unsed variable in case misusing.
del table

# Load features of images
for keyId in filteredFilelist:
    feature = db.getData(keyId)

    # If we found that the "unlabled image id" belong to a label.
    # Then skip it and do not put it into feature list.
    if feature is not None:
        if keyId == unlabeledImageID:
            continue

        featuresList.append(model.deserializeFeature(feature))

decompData = model.dimensionReduction(featuresList, decompFunction)

# Unlabeled Image Feature
if unlabeledImageID is not None:
    unlabelFeature = model.deserializeFeature(db.getData(unlabeledImageID))
elif unlabeledImagePath is not None:
    unlabelFeature = model.extractFeatures(cv2.imread(unlabeledImagePath))

if unlabelFeature is None:
    raise Exception("unlabeledImgFeature is None. Please check the unlabledImageID or unlabledImagePath is valid.")

# Flatten the feature to a vector so that we can use it to calculate the distance.
unlabelFeature = model.flattenFecture(unlabelFeature, decompMethod)

unlabelProjection = decompFunction.projectFeature(unlabelFeature, decompData, topk)

objLat = decompFunction.getObjLaten(decompData, topk)

# Any better way to calculate the distance?
for obj in objLat:
    print(distance(unlabelProjection, obj))
