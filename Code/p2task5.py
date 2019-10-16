from module.database import FilesystemDatabase
from module.dimensionReduction import DimReduction
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
from module.handMetadataParser import getFilelistByLabel
import cv2


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

if unlabeledImageID is None and unlabeledImagePath is None:
    parser.error('Please give unlabled image ID or path in -i / -ip argument.')

# Get filelist according to the label
filteredFilelist = getFilelistByLabel(metadataPath, label)

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
