import argparse
import os
import numpy as np
from pathlib import Path
from module.handMetadataParser import getFilelistByLabel
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase

parser = argparse.ArgumentParser(description="Phase 3 Task 1")

parser.add_argument(
    "-p",
    "--labeled_image_path",
    metavar="labeled_image_path",
    type=str,
    help="The folder path of labeled images.",
    required=True
)
parser.add_argument(
    "-unp",
    "--unlabeled_image_path",
    metavar="unlabeled_image_path",
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
labeledPath = Path(args.labeled_image_path)
unlabeledPath = Path(args.unlabeled_image_path)
k = args.k_latent
metadataPath = args.metadata

table = args.table.lower()
labeledFiles = []
unlabeledFiles = []

for fileName in os.listdir(labeledPath):
    if fileName.endswith(".jpg"):
        labeledFiles.append(fileName[:-4])
for fileName in os.listdir(unlabeledPath):
    if fileName.endswith(".jpg"):
        unlabeledFiles.append(fileName[:-4])
print(f"{len(labeledFiles)} labeled images and {len(unlabeledFiles)} unlabeled images.")

# get labels
palmarset = frozenset(getFilelistByLabel(metadataPath, 'p'))
palmarImage = []
dorsalImage = []
for img in labeledFiles:
    if img in palmarset:
        palmarImage.append(img)
    else:
        dorsalImage.append(img)

# computing latent semantics and projection
decompMethod = 'svd'
usedModels = ["cm", "lbp", "hog"] #task 1 uses these three models

db = FilesystemDatabase(f"{table}_cm", create=False)
model = modelFactory.creatModel("cm")
palmar_cm = []
for imgId in palmarImage:
    palmar_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
palmar_cm_latent = DimRed.createReduction(decompMethod, k=k, data=palmar_cm)
dorsal_cm = []
for imgId in dorsalImage:
    dorsal_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
dorsal_cm_latent = DimRed.createReduction(decompMethod, k=k, data=dorsal_cm)
unlabeled_cm = []
for imgId in unlabeledFiles:
    unlabeled_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
unlabeled_palmar_cm = palmar_cm_latent.transform(unlabeled_cm)
unlabeled_dorsal_cm = dorsal_cm_latent.transform(unlabeled_cm)

db = FilesystemDatabase(f"{table}_lbp", create=False)
model = modelFactory.creatModel("lbp")
palmar_lbp = []
for imgId in palmarImage:
    palmar_lbp.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
palmar_lbp_latent = DimRed.createReduction(decompMethod, k=k, data=palmar_lbp)
dorsal_lbp = []
for imgId in dorsalImage:
    dorsal_lbp.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
dorsal_lbp_latent = DimRed.createReduction(decompMethod, k=k, data=dorsal_lbp)
unlabeled_lbp = []
for imgId in unlabeledFiles:
    unlabeled_lbp.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
unlabeled_palmar_lbp = palmar_lbp_latent.transform(unlabeled_lbp)
unlabeled_dorsal_lbp = dorsal_lbp_latent.transform(unlabeled_lbp)

db = FilesystemDatabase(f"{table}_hog", create=False)
model = modelFactory.creatModel("hog")
palmar_hog = []
for imgId in palmarImage:
    palmar_hog.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
palmar_hog_latent = DimRed.createReduction(decompMethod, k=k, data=palmar_hog)
dorsal_hog = []
for imgId in dorsalImage:
    dorsal_hog.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
dorsal_hog_latent = DimRed.createReduction(decompMethod, k=k, data=dorsal_hog)
unlabeled_hog = []
for imgId in unlabeledFiles:
    unlabeled_hog.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
unlabeled_palmar_hog = palmar_hog_latent.transform(unlabeled_hog)
unlabeled_dorsal_hog = dorsal_hog_latent.transform(unlabeled_hog)

# combining latent projection and labeling
cmHit = 0
lbpHit = 0
hogHit = 0
length = len(unlabeledFiles)

for index, imgId in enumerate(unlabeledFiles):
    isPalmar = imgId in palmarset
       
    if isPalmar == (np.linalg.norm(unlabeled_palmar_cm[index]) > np.linalg.norm(unlabeled_dorsal_cm[index])):
        cmHit+=1
    if isPalmar == (np.linalg.norm(unlabeled_palmar_lbp[index]) > np.linalg.norm(unlabeled_dorsal_lbp[index])):
        lbpHit+=1
    if isPalmar == (np.linalg.norm(unlabeled_palmar_hog[index]) > np.linalg.norm(unlabeled_dorsal_hog[index])):
        hogHit+=1
print(f"cm hit rate is {cmHit/length}; lbp hit rate is {lbpHit/length}; hog hit rate is {hogHit/length}")    
