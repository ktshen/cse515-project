from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
from pathlib import Path
import argparse
from module.handMetadataParser import getFilelistByLabel


parser = argparse.ArgumentParser(description="Phase 2 Task 3")
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
    "-p",
    "--image_path",
    metavar="image_path",
    type=str,
    help="The folder path of images.",
    required=True,
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
args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk
decompMethod = args.method.lower()
metadataPath = args.metadata
imagePath = Path(args.image_path)
label = args.label   # Used to filter such as left-hand or right-hand

# Get all image id with the specified label
filteredFilelist = getFilelistByLabel(metadataPath, label)

# Open database
db = FilesystemDatabase(f"{table}_{model}", create=False)

# Create model
model = modelFactory.creatModel(model)
# The imageIdList and featureList should could be mapped to each other.
objFeat = []
objId = []

# Removed unsed variable in case misusing.
del table

# Load features of images
for keyId in filteredFilelist:
    feature = db.getData(keyId)

    # Because the database may store subset of the whole dataset.
    # Some keyId may not exsit in database.
    if feature is not None:
        objId.append(keyId)
        objFeat.append(
            model.flattenFecture(model.deserializeFeature(feature), decompMethod)
        )   
# Create latent semantics
latentModel = DimRed.createReduction(decompMethod, k=topk, data=objFeat)
# Output results
latentModel.printLatentSemantics(objId, objFeat, imagePath)