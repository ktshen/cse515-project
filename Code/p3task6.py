import argparse
import numpy as np
import pickle
from classifier.classifier import Classifier
from module.DimRed import DimRed
from module.database import FilesystemDatabase
from models import modelFactory
from module.PRF import prf
import sys


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
    help="The model of image feature will be used.",
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
allImageData = []

for cand in candidates:
    feature = model.flattenFecture(
        model.deserializeFeature(db.getData(cand[0])), decompMethod
    )
    imageDataDict[cand[0]] = feature
    allImageData.append(feature)

allImageData = np.array(allImageData)

# Label images
print("Label the images as r(relevant), i(irrelevant) or ?(unknown)")

retrainData = []
retrainLabel = []
retrainFileID = []

rFileIDList = []
irFileIDList = []
allFileIDList = []
rIrFileIDList = []

for cand in candidates:
    while True:
        label = input(f"{cand[0]} > ").strip()
        allFileIDList.append(cand[0])

        if label == "r":
            retrainData.append(imageDataDict[cand[0]])
            retrainLabel.append(True)
            rFileIDList.append(cand[0])
            rIrFileIDList.append(cand[0])
            break
        elif label == "i":
            retrainData.append(imageDataDict[cand[0]])
            retrainLabel.append(False)
            irFileIDList.append(cand[0])
            rIrFileIDList.append(cand[0])
            break
        elif label == "?":
            break

if args.classifier.lower() == "prf":
    print("The reordered results are:")
    print(prf(rFileIDList, irFileIDList, allFileIDList, table))

    # PRF do not need to run the following code.
    sys.exit(0)

if len(retrainData) == 0:
    print("Please give some relevant and irrelevant label")

print("Doing dimension reduction on training data...")
if decompMethod is not None and topk is not None:
    # Create latent semantics
    latentModel = DimRed.createReduction(decompMethod, k=topk, data=retrainData)
    # Transform data
    retrainData = latentModel.transform(retrainData)

# Training classifier
print("Training classifier by training data...")
retrainData = np.array(retrainData)


if args.classifier.lower() == "ppr":
    image_list = rIrFileIDList
    image_simlarity_dict = {}

    for i, feature in enumerate(retrainData):
        img_img_sim = []
        for feature_p in retrainData:
            img_img_sim.append(np.dot(feature, feature_p.T))

        a = np.array(img_img_sim)
        ind = a.argsort()[-20:][::-1]

        image_simlarity_dict[image_list[i]] = {
            "sim_weight": a[ind],
            "sim_node_index": ind,
        }
    classifier = Classifier.createClassifier(
        args.classifier.lower(),
        **{"img_sim_graph": image_simlarity_dict, "image_list": image_list},
    )
else:
    classifier = Classifier.createClassifier(args.classifier.lower())

classifier.fit(retrainData, retrainLabel)


print("Doing dimension reduction on all images data...")
if decompMethod is not None and topk is not None:
    # Transform data
    allImageData = latentModel.transform(allImageData)


print("Predict all Image data by classifier")
allImagePredictResult = classifier.predict(allImageData)

relevantImages = []
irrelevantImages = []

for i in range(len(allImagePredictResult)):
    if allImagePredictResult[i]:
        relevantImages.append(candidates[i][0])
    else:
        irrelevantImages.append(candidates[i][0])

print("Relevant image:")
for r in relevantImages:
    print(r)

print("Irrelevant images:")
for r in irrelevantImages:
    print(r)