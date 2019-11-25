import argparse
import numpy as np
import os
import pickle
from classifier.classifier import Classifier
from module.DimRed import DimRed
from pathlib import Path
from module.database import FilesystemDatabase
from models import modelFactory


parser = argparse.ArgumentParser(description="Phase 3 Task 6")

parser.add_argument(
    "-c",
    "--classifier",
    metavar="classifier",
    type=str,
    help="Classifier",
    required=True,
)
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
parser.add_argument(
    "-d",
    "--method",
    metavar="method",
    type=str,
    help="The method will be used to reduce dimension.",
    required=False,
)
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=False)


# extract argument
args = parser.parse_args()

classifier = Classifier.createClassifier(args.classifier.lower())
modelName = args.model.lower() if args.model else None
table = args.table.lower() if args.table else None

decompMethod = args.method.lower() if args.method else None
topk = args.topk


# Read the output of task 5
with open("task5_output.pkl", "rb") as fHndl:
    queryImage, candidates = pickle.load(fHndl)


# Open database
db = FilesystemDatabase(f"{table}_{modelName}", create=False)
model = modelFactory.creatModel(modelName)

imageDataDict = {}

for cand in candidates:
    imageDataDict[cand[0]] = model.flattenFecture(
        model.deserializeFeature(db.getData(cand[0])), decompMethod
    )

# Label images
print("Label the images as r(relevant), i(irrelevant) or ?(unknown)")

retrainData = []
retrainLabel = []
retrainFileID = []

for cand in candidates:
    while True:
        label = input(f"{cand[0]} > ").strip()
        if label == "r":
            retrainData.append(imageDataDict[cand[0]])
            retrainLabel.append(True)
            break
        elif label == "i":
            retrainData.append(imageDataDict[cand[0]])
            retrainLabel.append(False)
            break
        elif label == "?":
            break

if decompMethod is not None and topk is not None:
    print("Doing dimension reduction on training data...")
    # Create latent semantics
    latentModel = DimRed.createReduction(decompMethod, k=topk, data=retrainData)
    # Transform data
    retrainData = latentModel.transform(retrainData)

# Training classifier
print("Training Classifer by training data...")
retrainData = np.array(retrainData)
classifier.fit(retrainData, retrainLabel)

