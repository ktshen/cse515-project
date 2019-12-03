import argparse
import os
import numpy as np
from pathlib import Path
from module.handMetadataParser import getFilelistByLabelP3
from module.handMetadataParser import getFilelistByLabel
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase

# Processing arguments
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

parser.add_argument(
    "-test",
    "--test",
    metavar="test",
    type=bool,
    help="test mode.",
    required=False,
    default=False
)

args = parser.parse_args()
labeledPath = Path(args.labeled_image_path)
unlabeledPath = Path(args.unlabeled_image_path)
k = args.k_latent
metadataPath = args.metadata
table = args.table.lower()
istest = args.test

# Get labeled and unlabeled images' IDs
labeledFiles = []
unlabeledFiles = []
for fileName in os.listdir(labeledPath):
    if fileName.endswith(".jpg"):
        labeledFiles.append(fileName[:-4])
for fileName in os.listdir(unlabeledPath):
    if fileName.endswith(".jpg"):
        unlabeledFiles.append(fileName[:-4])
print(f"{len(labeledFiles)} labeled images and {len(unlabeledFiles)} unlabeled images.")

# Separate palmar and dorsal images
if istest:
    palmarset = frozenset(getFilelistByLabel(metadataPath, 'p'))
else:
    palmarset = frozenset(getFilelistByLabelP3(metadataPath, 'p'))
    
palmarImage = []
dorsalImage = []
for img in labeledFiles:
    if img in palmarset:
        palmarImage.append(img)
    else:
        dorsalImage.append(img)

"""
The approach is according to the vector's P-norm in SVD latent space.
We will first measure their accuracies individually and 
then try to combine them to get a better results.
"""
decompMethod = 'svd'
usedModels = ["cavg", "hog"] 

# Read color average descriptors and compute latent semantics
db = FilesystemDatabase(f"{table}_cavg", create=False)
model = modelFactory.creatModel("cavg")

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
    
# Projection    
unlabeled_palmar_cm = palmar_cm_latent.transform(unlabeled_cm)
unlabeled_dorsal_cm = dorsal_cm_latent.transform(unlabeled_cm)
  
# Read hog descriptors and compute latent semantics
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
    
# Projection        
unlabeled_palmar_hog = palmar_hog_latent.transform(unlabeled_hog)
unlabeled_dorsal_hog = dorsal_hog_latent.transform(unlabeled_hog)
  
# we use pnorm to measure vector length, then compare length 
pNorm = 2
CMfactor = 60
HOGfactor = 1
if istest:
    overallAccuracy = 0     #overall correct prediction
    CMAccuracy = 0          #correct prediction using CM
    HOGAccuracy = 0         #correct prediction using HOG
    totalNum = len(unlabeledFiles)
else:
    palmar = []
    dorsal = []

for index, imgId in enumerate(unlabeledFiles):
    biasToP = 0
      
    cmPlen = np.linalg.norm(unlabeled_palmar_cm[index], ord = pNorm) 
    cmDlen = np.linalg.norm(unlabeled_dorsal_cm[index], ord = pNorm)
    CMdiff = (cmPlen - cmDlen) / max(cmPlen, cmDlen)        
  
    hogPlen = np.linalg.norm(unlabeled_palmar_hog[index], ord = pNorm) 
    hogDlen = np.linalg.norm(unlabeled_dorsal_hog[index], ord = pNorm)
    HOGdiff = (hogPlen - hogDlen) / max(hogPlen, hogDlen)    
  
    biastoP = CMdiff * CMfactor + HOGdiff * HOGfactor
            
    if istest:  
        isPalmar = imgId in palmarset
        if isPalmar == (CMdiff > 0):
            CMAccuracy += 1 
        if isPalmar == (HOGdiff > 0):
            HOGAccuracy += 1
        if isPalmar == (biastoP > 0):
            overallAccuracy += 1 
    else:
        if biastoP > 0:
            palmar.append(imgId)
        else:
            dorsal.append(imgId) 
              
if istest:          
    print(f"Overall accuracy is {overallAccuracy/totalNum}")
    print(f"CM accuracy is {CMAccuracy/totalNum}; HOG accuracy is {HOGAccuracy/totalNum}")
else:
    print("Images labeled as palmar are:")
    print(palmar)
    print("Images labeled as dorsal are:")
    print(dorsal)
    