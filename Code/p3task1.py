import argparse
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from module.handMetadataParser import getFilelistByLabel
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase

parser = argparse.ArgumentParser(description="Phase 3 Task 1")

parser.add_argument(
    "-p",
    "--labeled_image_path",
    metavar="labeled_path",
    type=str,
    help="The folder path of labeled images.",
    required=True
)
parser.add_argument(
    "-unp",
    "--unlabeled_image_path",
    metavar="unlabeled_path",
    type=str,
    help="The folder path of unlabeled images.",
    required=True
)
parser.add_argument(
    "-k", 
    "--k_latent", 
    metavar="k", 
    type=int, 
    help="k latent semantics", 
    required=True
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
    "-meta",
    "--metadata",
    metavar="METADATA_PATH",
    type=str,
    help="Path of metadata.",
    required=True
)

args = parser.parse_args()
labeledPath = Path(args.labeled_path)
unlabeledPath = Path(args.unlabeled_path)
k = args.k
metadataPath = args.metadata

table = args.table.lower()
labeledFiles = []
unlabeledFiles = []

for fileName in os.listdir(labeledPath):
    if fileName.endswith(".jpg"):
        labeledFiles.append(labeledPath / fileName)
for fileName in os.listdir(unlabeledPath):
    if fileName.endswith(".jpg"):
        unlabeledFiles.append(unlabeledPath / fileName)
print(f"{len(labeledFiles)} labeled images and {len(unlabeledFiles)} unlabeled images.")

# get labels
palmarset = frozenset(getFilelistByLabel(metadataPath, 'p'))
palmarImage = []
doesalImage = []
for img in labeledFiles:
    if img in palmarset:
        palmarImage.append(img)
    else:
        doesalImage.append(img)

# TODO processing and labeling

