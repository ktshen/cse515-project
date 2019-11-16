import argparse
import os
import numpy as np
from pathlib import Path
from module.handMetadataParser import getFilelistByLabelP3
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase
from sklearn.metrics.pairwise import cosine_similarity

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
palmarset = frozenset(getFilelistByLabelP3(metadataPath, 'p'))
palmarImage = []
dorsalImage = []
for img in labeledFiles:
    if img in palmarset:
        palmarImage.append(img)
    else:
        dorsalImage.append(img)


"""
The first approach is according to the vector's P-norm in SVD latent space.
We will first measure their accuracies individually and 
then try to combine them to get a better results.
"""

# computing latent semantics and projection
decompMethod = 'svd'
usedModels = ["cavg", "hog"] #task 1 uses these three models
  
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
    
unlabeled_palmar_cm = palmar_cm_latent.transform(unlabeled_cm)
unlabeled_dorsal_cm = dorsal_cm_latent.transform(unlabeled_cm)
  
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
  
# we use pnorm to measure vector length, then compare length 
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
      
    cmPlen = np.linalg.norm(unlabeled_palmar_cm[index], ord = pNorm) 
    cmDlen = np.linalg.norm(unlabeled_dorsal_cm[index], ord = pNorm)
    CMdiff = (cmPlen - cmDlen) / max(cmPlen, cmDlen)
#     if isPalmar == (CMdiff > 0):
#         CMRate += 1     
  
    hogPlen = np.linalg.norm(unlabeled_palmar_hog[index], ord = pNorm) 
    hogDlen = np.linalg.norm(unlabeled_dorsal_hog[index], ord = pNorm)
    HOGdiff = (hogPlen - hogDlen) / max(hogPlen, hogDlen)
#     if isPalmar == (HOGdiff > 0):
#         HOGRate += 1
  
    biastoP = CMdiff * CMfactor + HOGdiff * HOGfactor
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

"""
The second approach is according to the vector's cosine similarity to SVD latent semantics.
We will first measure their accuracies individually and 
then try to combine them to get a better results. After doing some experiments, this method does not perfrom well.
"""

# # computing latent semantics and projection
# decompMethod = 'svd'
# usedModels = ["cavg", "lbp", "hog"] #task 1 uses these three models
#  
# db = FilesystemDatabase(f"{table}_cavg", create=False)
# model = modelFactory.creatModel("cavg")
# palmar_cm = []
# for imgId in palmarImage:
#     palmar_cm.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
# CMP = DimRed.createReduction(decompMethod, k=k, data=palmar_cm)
# palmar_cm_latent = CMP.getLatentSemantics()
# palmar_cm_weight = CMP.getWeights()
# dorsal_cm = []
# for imgId in dorsalImage:
#     dorsal_cm.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
# CMD = DimRed.createReduction(decompMethod, k=k, data=dorsal_cm)
# dorsal_cm_latent = CMD.getLatentSemantics()
# dorsal_cm_weight = CMD.getWeights()
# unlabeled_cm = []
# for imgId in unlabeledFiles:
#     unlabeled_cm.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
#  
# db = FilesystemDatabase(f"{table}_lbp", create=False)
# model = modelFactory.creatModel("lbp")
# palmar_lbp = []
# for imgId in palmarImage:
#     palmar_lbp.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
# LBPP = DimRed.createReduction(decompMethod, k=k, data=palmar_lbp)
# palmar_lbp_latent = LBPP.getLatentSemantics()
# palmar_lbp_weight = LBPP.getWeights()
# dorsal_lbp = []
# for imgId in dorsalImage:
#     dorsal_lbp.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
# LBPD = DimRed.createReduction(decompMethod, k=k, data=dorsal_lbp)
# dorsal_lbp_latent = LBPD.getLatentSemantics()
# dorsal_lbp_weight = LBPD.getWeights()
# unlabeled_lbp = []
# for imgId in unlabeledFiles:
#     unlabeled_lbp.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
#  
# db = FilesystemDatabase(f"{table}_hog", create=False)
# model = modelFactory.creatModel("hog")
# palmar_hog = []
# for imgId in palmarImage:
#     palmar_hog.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
# HOGP = DimRed.createReduction(decompMethod, k=k, data=palmar_hog)
# palmar_hog_latent = HOGP.getLatentSemantics()
# palmar_hog_weight = HOGP.getWeights()
# dorsal_hog = []
# for imgId in dorsalImage:
#     dorsal_hog.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
# HOGD = DimRed.createReduction(decompMethod, k=k, data=dorsal_hog)
# dorsal_hog_latent = HOGD.getLatentSemantics()
# dorsal_hog_weight = HOGD.getWeights()
# unlabeled_hog = []
# for imgId in unlabeledFiles:
#     unlabeled_hog.append(
#         model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)
#     )
#  
# # we use cosine similarity to measure
# overallRate = 0     #overall correct prediction
# CMRate = 0          #correct prediction using CM
# LBPRate = 0         #correct prediction using LBP
# HOGRate = 0         #correct prediction using HOG
#  
# noComp = 0          #all models predict wrongly
#  
# totalNum = len(unlabeledFiles)
# CMfactor = 1
# LBPfactor = 1
# HOGfactor = 1
#  
# for imgIdx in range(len(unlabeledFiles)):
#     isPalmar = unlabeledFiles[imgIdx] in palmarset
#      
#     cosineSimCMP = 0
#     cosineSimCMD = 0    
#     for ls in zip(palmar_cm_latent, palmar_cm_weight):
#         cosineSimCMP += cosine_similarity(np.array(unlabeled_cm[imgIdx]).reshape(1, -1), np.array(ls[0]).reshape(1, -1))[0][0] * ls[1]
#     for ls in zip(dorsal_cm_latent, dorsal_cm_weight):
#         cosineSimCMD += cosine_similarity(np.array(unlabeled_cm[imgIdx]).reshape(1, -1), np.array(ls[0]).reshape(1, -1))[0][0] * ls[1]      
#     if isPalmar == (cosineSimCMP > cosineSimCMD):
#         CMRate += 1
#     CMdiff = cosineSimCMP - cosineSimCMD
#     
#     cosineSimLBPP = 0
#     cosineSimLBPD = 0
#     for ls in palmar_lbp_latent:
#         cosineSimLBPP += cosine_similarity(np.array(unlabeled_lbp[imgIdx]).reshape(1, -1), np.array(ls).reshape(1, -1))[0][0]
#     for ls in dorsal_lbp_latent:
#         cosineSimLBPD += cosine_similarity(np.array(unlabeled_lbp[imgIdx]).reshape(1, -1), np.array(ls).reshape(1, -1))[0][0]        
#     if isPalmar == (cosineSimLBPP > cosineSimLBPD):
#         LBPRate += 1
#     LBPdiff = cosineSimLBPP - cosineSimLBPD
#      
#     cosineSimHOGP = 0
#     cosineSimHOGD = 0    
#     for ls in palmar_hog_latent:
#         cosineSimHOGP += cosine_similarity(np.array(unlabeled_hog[imgIdx]).reshape(1, -1), np.array(ls).reshape(1, -1))[0][0]
#     for ls in dorsal_hog_latent:
#         cosineSimHOGD += cosine_similarity(np.array(unlabeled_hog[imgIdx]).reshape(1, -1), np.array(ls).reshape(1, -1))[0][0]        
#     if isPalmar == (cosineSimHOGP > cosineSimHOGD):
#         HOGRate += 1
#     HOGdiff = cosineSimHOGP - cosineSimHOGD
#  
#  
#     biastoP = CMdiff * CMfactor + LBPdiff * LBPfactor + HOGdiff * HOGfactor
#     if isPalmar == (biastoP > 0):
#         overallRate += 1
#     else:
#         if (CMdiff > 0 and LBPdiff > 0 and HOGdiff > 0) or (CMdiff < 0 and LBPdiff < 0 and HOGdiff < 0):
#             noComp += 1
#         else:
#             print(f"the CMdiff is {CMdiff}; the LBPdiff is {LBPdiff}; the HOG is {HOGdiff}")
#          
# print(f"Overall hit rate is {overallRate/totalNum}")
# print(f"CM rate is {CMRate/totalNum}; LBP rate is {LBPRate/totalNum}; HOG rate is {HOGRate/totalNum}")
# print(f"no compensate misses {noComp/(totalNum - overallRate)}")
