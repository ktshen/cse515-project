import argparse
import numpy as np
from module.handMetadataParser import getFilelistWithOneHot
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase


# Get necessary arguments
parser = argparse.ArgumentParser(description="Phase 2 Task 8")
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=True)
parser.add_argument(
    "-meta",
    "--metadata",
    metavar="METADATA_PATH",
    type=str,
    help="Path of metadata.",
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    metavar="model",
    type=str,
    help="The model will be used.",
    default="lbp",
)
parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=True,
)
parser.add_argument(
    "-d",
    "--method",
    metavar="method",
    type=str,
    help="The method will be used to reduce dimension.",
    default="svd",
)

# extract argument
args = parser.parse_args()
topk = args.topk
metadataPath = args.metadata
modelName = args.model.lower()
decompMethod = args.method.lower()
table = args.table.lower()

# Open database according to model and table name
db = FilesystemDatabase(f"{table}_{modelName}", create=False)
# Instantiate model
model = modelFactory.creatModel(modelName)

objFeat = []
objId = []

# Load data from database
print("Loading data from database")
for keyId in db.keys():
    objId.append(keyId)
    objFeat.append(
        model.flattenFecture(model.deserializeFeature(db.getData(keyId)), decompMethod)
    )

# Create latent semantics
print("Doing Dimension Reduction")
latentModel = DimRed.createReduction(decompMethod, k=topk, data=objFeat)
imageSpace = latentModel.transform(objFeat)

imgSpaceMat = np.array(imageSpace)

# Load onehot vector
imgToOneHot = getFilelistWithOneHot(metadataPath)

# Shape of image space and onehot vector
imgMetaShape = (imageSpace[0].shape[0], list(imgToOneHot.values())[0].shape[0])

# Create binary image-metadata matrix
imgMetaMat = []

print("Building a binary image-metadata matrix")
for i, keyId in enumerate(objId):
    imgSpace = objFeat[i]
    onehot = imgToOneHot[keyId]
    newFeature = np.concatenate((imgSpace, onehot))
    imgMetaMat.append(newFeature)

print("Doing NMF with image-metadata matrix")
nmf = DimRed.createReduction("nmf", k=topk, data=imgMetaMat)

print("Top-k latent semantics in the image-space -> (order, term)")
for order, array in enumerate(nmf.components_):
    print((order + 1, array[: imgMetaShape[0]]))

print("Top-k latent semantics in the metadata-space -> (order, term)")
for order, array in enumerate(nmf.components_):
    print((order + 1, array[imgMetaShape[0]:]))
