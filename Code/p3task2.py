import argparse
import os
import numpy as np
from pathlib import Path
from module.handMetadataParser import getFilelistByLabelP3
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description="Phase 3 Task 2")

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
    "-c", 
    "--c_cluster", 
    metavar="c", 
    type=int, 
    help="The number of clusters", 
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
c = args.c_cluster
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
palmarset = frozenset(getFilelistByLabelP3(metadataPath, 'p'))
palmarImage = []
dorsalImage = []
for img in labeledFiles:
    if img in palmarset:
        palmarImage.append(img)
    else:
        dorsalImage.append(img)


"""

"""

# compute clusters and find representatives
decompMethod = 'svd'
usedModels = ["cavg", "hog"] 
  
db = FilesystemDatabase(f"{table}_cavg", create=False)
model = modelFactory.creatModel("cavg")

palmar_cm = []
for imgId in palmarImage:
    palmar_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )

dorsal_cm = []
for imgId in dorsalImage:
    dorsal_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )

unlabeled_cm = []
for imgId in unlabeledFiles:
    unlabeled_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
  
db = FilesystemDatabase(f"{table}_hog", create=False)
model = modelFactory.creatModel("hog")

palmar_hog = []
for imgId in palmarImage:
    palmar_hog.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )

dorsal_hog = []
for imgId in dorsalImage:
    dorsal_hog.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )

unlabeled_hog = []
for imgId in unlabeledFiles:
    unlabeled_hog.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
    
  

overallRate = 0     #overall correct prediction
CMRate = 0          #correct prediction using CM
HOGRate = 0         #correct prediction using HOG
noComp = 0          #all models predict wrongly
  
totalNum = len(unlabeledFiles)
pNorm = 2
CMfactor = 50
HOGfactor = 1
  
for index, imgId in enumerate(unlabeledFiles):
#     isPalmar = imgId in palmarset
    biasToP = 0
      
    cmPlen = 0
    cmDlen = 0
    CMdiff = 0
#     if isPalmar == (CMdiff > 0):
#         CMRate += 1     
  
    hogPlen = 0
    hogDlen = 0
    HOGdiff = 0
#     if isPalmar == (HOGdiff > 0):
#         HOGRate += 1
  
    biastoP = 0
#     if isPalmar == (biastoP > 0):
#         overallRate += 1
#     else:
#         if (CMdiff > 0 and HOGdiff > 0) or (CMdiff < 0 and HOGdiff < 0):
#             noComp += 1
#         else:
#             print(f"the CMdiff is {CMfactor * CMdiff}; the HOG is {HOGfactor * HOGdiff}")
          
# print(f"Overall hit rate is {overallRate/totalNum}")
# print(f"CM rate is {CMRate/totalNum}; HOG rate is {HOGRate/totalNum}")
# print(f"no compensate misses {noComp/(totalNum - overallRate)}")
    if biastoP > 0:
        print(f"{imgId} is Palmar")
    else:
        print(f"{imgId} is Dorsal")
