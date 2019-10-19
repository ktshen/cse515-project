from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
from module.handMetadataParser import getFilelistByID
import cv2
import itertools
from Pathlib import Path
from collections import defaultdict


parser = argparse.ArgumentParser(description="Phase 2 Task 6")
parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=True,
)

parser.add_argument(
    "-s",
    "--subject_id",
    metavar="subject_id",
    type=str,
    help="The subject image ID.",
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
table = args.table.lower()
subjectID = args.subject_id
metadataPath = Path(args.metadata)

# Get all models and dimention reduction methods
models = modelFactory.getSupportModel()
decompMethods = DimRed.getSupportedMethods()

# Two dict,
# subToImg: subject ID -> image ID
# imgToSub: image ID -> subject ID
subToImg, imgToSub = getFilelistByID()


# TODO: Test LBP First!!!!!

# Test LBP first
db = FilesystemDatabase(f"{table}_lbp", create=False)
model = modelFactory.creatModel()

# subect ID -> features
subjectFeatures = {}

for keyId in db.keys():
    subjectFeatures[keyId].append()
