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

# (model name, dimension reduction name) -> feature
subjectFeature = {}

# model with corressponding latent space.
# (model name, dimension reduction name) -> DimRed instance
latents = {}

previousModelName = None

productList = list(itertools.product(models, decompMethods))

defaultTopK = 20

for i, p in enumerate(productList):

    modelName, decompMethod = p

    print(f"Calculating: {i} / {len(productList)} - {modelName.upper()} with {decompMethod.upper()}")

    if modelName != previousModelName:
        db = FilesystemDatabase(f"{table}_{modelName}", create=False)
        model = modelFactory.creatModel(modelName)

    featureMatrix = []

    for keyId in db.keys():
        feature = model.flattenFecture(model.deserializeFeature(db.getData(keyId)), decompMethod)

        if keyId == subjectImageID:
            subjectFeature[(modelName, decompMethod)] = feature
        else:
            featureMatrix.append(feature)

    latentData = DimRed.createReduction(decompMethod, k=defaultTopK, data=featureMatrix)
    latents[(modelName, decompMethod)] = latentData

# Removed unused variable in case misusing.
del table
del productList
del previousModelName
del defaultTopK
del modelName
del decompMethod
del p
del i
del featureMatrix

# TODO: Compare the feature of subject image with others
