import sys
from module.database import FilesystemDatabase
from module.distance import Norm
from module.dimensionReduction import DimReduction
from pathlib import Path
import cv2
from models import modelFactory
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Phase 2 Task 1")
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
args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk
method = args.method.lower()

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}", create=False)
model = modelFactory.creatModel(model)
method = DimReduction.createReduction(method, k=topk)

# Removed unsed variable in case misusing.
del table

featuresList = []

# Load features of images
for keyId in db.keys():
    featuresList.append(model.deserializeFeature(db.getData(keyId)))

U, s, V = model.dimensionReduction(featuresList, method)

# TODO: How to print the original latent vector
print(s)