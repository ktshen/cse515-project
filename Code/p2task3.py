from module.database import FilesystemDatabase
from module.dimensionReduction import DimReduction
from models import modelFactory
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
args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk
method = args.method.lower()
distFunction = args.distance.lower()
metadataPath = args.metadata
label = args.label   # Used to filter such as left-hand or right-hand

# Get filelist according to the label
filteredFilelist = getFilelistByLabel(metadataPath, label)

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}", create=False)

model = modelFactory.creatModel(model)
method = DimReduction.createReduction(method, k=topk)

# The imageIdList and featureList should could be mapped to each other.
featuresList = []

# Removed unsed variable in case misusing.
del table

# Load features of images
for keyId in filteredFilelist:
    feature = db.getData(keyId)

    # Because the database may store subset of the whole dataset.
    # Some keyId may not exsit in database.
    if feature is not None:
        featuresList.append(model.deserializeFeature(feature))

_, termWeight = model.dimensionReduction(featuresList, method)

# TODO: How to print the original latent vector
print(termWeight)
