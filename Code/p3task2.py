import argparse
import os
import numpy as np
from pathlib import Path
from module.handMetadataParser import getFilelistByLabel
from module.handMetadataParser import getFilelistByLabelP3
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase
from module.kmeans import KMeans

# Processing arguments
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
parser.add_argument(
    "-test",
    "--test",
    metavar="test",
    type=bool,
    help="test mode.",
    required=False,
    default=False
)
parser.add_argument(
    "-k",
    "--k",
    metavar="k",
    type=int,
    help="the number of latent semantics.",
    required=False,
    default=10
)

args = parser.parse_args()
labeledPath = Path(args.labeled_image_path)
unlabeledPath = Path(args.unlabeled_image_path)
c = args.c_cluster
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
In this method, we do clustering in PCA latent semantics. The length is measured by L1-norm.
"""
# compute clusters and find representatives
k = args.k
decompMethod = 'pca'
usedModels = ["cavg"] 

# Read color average descriptors     
db = FilesystemDatabase(f"{table}_cavg", create=False)
model = modelFactory.creatModel("cavg")
   
palmar_cm = []
for imgId in palmarImage:
    palmar_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
# Compute palmar latent semantics
palmar_cm_latent = DimRed.createReduction(decompMethod, k=k, data=palmar_cm)
palmar_cm = palmar_cm_latent.transform(palmar_cm)

# Compute palmar clustering
palmar_cm_cluster = KMeans(n_clusters=c).fit(np.array(palmar_cm))
palmar_cm_centroid = palmar_cm_cluster.cluster_centers_ 
palmar_clusters = [[] for i in range(c) ]
for i, l in enumerate(palmar_cm_cluster.labels_):
    palmar_clusters[l].append(palmarImage[i])
   
dorsal_cm = []
for imgId in dorsalImage:
    dorsal_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
    
# Compute palmar latent semantics
dorsal_cm_latent = DimRed.createReduction(decompMethod, k=k, data=dorsal_cm)
dorsal_cm = dorsal_cm_latent.transform(dorsal_cm)

# Compute palmar clustering
dorsal_cm_cluster = KMeans(n_clusters=c).fit(np.array(dorsal_cm))
dorsal_cm_centroid = dorsal_cm_cluster.cluster_centers_ 
dorsal_clusters = [[] for i in range(c) ]
for i, l in enumerate(dorsal_cm_cluster.labels_):
    dorsal_clusters[l].append(dorsalImage[i])
 
# Projection   
unlabeled_cm = []
for imgId in unlabeledFiles:
    unlabeled_cm.append(
        model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
    )
unlabeled_palmar_cm = palmar_cm_latent.transform(unlabeled_cm)
unlabeled_dorsal_cm = dorsal_cm_latent.transform(unlabeled_cm)                
  
pNorm = 1
if istest:
    overallAccuracy = 0     #overall correct prediction
    totalNum = len(unlabeledFiles)   
else:
    palmar = []
    dorsal = []
    print("The dorsal clusters are:")
    for images in dorsal_clusters:
        print(images)
    print("The palmar clusters are:")
    for images in palmar_clusters:
        print(images)  
             
for index, imgId in enumerate(unlabeledFiles):         
    cmPlen = 0
    cmDlen = 0
    for idx, centroid in enumerate(palmar_cm_centroid):
        cmPlen += len(palmar_clusters[idx]) * np.linalg.norm(centroid - unlabeled_palmar_cm[index], ord = pNorm)
   
    for idx, centroid in enumerate(dorsal_cm_centroid):
        cmDlen += len(dorsal_clusters[idx]) * np.linalg.norm(centroid - unlabeled_dorsal_cm[index], ord = pNorm)    
        
    if istest:  
        isPalmar = imgId in palmarset
        if isPalmar == (cmPlen < cmDlen):
            overallAccuracy += 1  
    else:
        if cmPlen < cmDlen:
            palmar.append(imgId)
        else:
            dorsal.append(imgId)  
          
if istest:          
    print(f"Overall accuracy is {overallAccuracy/totalNum}")
else:
    print("Images labeled as palmar are:")
    print(palmar)
    print("Images labeled as dorsal are:")
    print(dorsal)
