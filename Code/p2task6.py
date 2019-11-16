from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
import argparse
from module.distanceFunction import distanceFunction
import sklearn.metrics.pairwise as similarity
from module.handMetadataParser import getFilelistByID
import cv2
import itertools
from pathlib import Path
from collections import defaultdict
import numpy as np

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

parser.add_argument(
    "-p",
    "--image_path",
    metavar="image_path",
    type=str,
    help="The folder path of images.",
    required=True,
)

args = parser.parse_args()

# extract argument
table = args.table.lower()
subjectID = args.subject_id
metadataPath = Path(args.metadata)
imagePath = args.image_path

# Get all models and dimention reduction methods
models = modelFactory.getSupportModel()
decompMethods = DimRed.getSupportedMethods()

# Two dict,
# subToImg: subject ID -> image ID
# imgToSub: image ID -> subject ID
subToImg, imgToSub = getFilelistByID(metadataPath)


decompMethod = 'lda'
smallest_k = 99999


db = FilesystemDatabase(f"{table}_cm", create=False)
model = modelFactory.creatModel('cavg')

# The imageIdList and featureList should could be mapped to each other.
target_sub_feat = []
target_subject_img_lst =  subToImg.get(int(subjectID))
subject_img_dic_pool = {}

if target_subject_img_lst is None:
    print("Error! There has no such subject ID: ", subjectID)
    exit(1)


for f_name in target_subject_img_lst:
    feature = db.getData(f_name)
    if feature is not None:
        target_sub_feat.append(
            model.flattenFecture(model.deserializeFeature(feature), decompMethod)
        )
if not target_sub_feat:
    print("Error! Could not find any images in the database with subject ID: ", subjectID)
    exit(1)



for key in db.keys():
    subject_id = imgToSub.get(key)

    if int(subject_id) == int(subjectID):
        continue
    elif subject_img_dic_pool.get(subject_id) is None:
        subject_img_dic_pool[subject_id] = {
            'image_list' : [key],
            'flatten_feature':[],
            'latentModel':""
        }
    else:
        (subject_img_dic_pool.get(subject_id)['image_list']).append(key)



if smallest_k>len(target_sub_feat):
    smallest_k = len(target_sub_feat)


for key, value in subject_img_dic_pool.items():
    temp_feature = []
    for f_name in value['image_list']:
        feature = db.getData(f_name)
        if feature is not None:
            temp_feature.append(
                model.flattenFecture(model.deserializeFeature(feature), decompMethod)
            )
    value['flatten_feature'] = temp_feature
    if smallest_k > len(value['flatten_feature']):
        smallest_k = len(value['flatten_feature'])

target_latent_model = DimRed.createReduction(decompMethod, k=smallest_k, data=target_sub_feat)
flattend_target_latent_model = np.reshape(target_latent_model.getLatentSemantics(),(1,-1))


simPair = []
for key,value in subject_img_dic_pool.items():
    value['latentModel'] = DimRed.createReduction(decompMethod, k= smallest_k, data=value['flatten_feature'] )
    flattend_target_latent_pool_model = np.reshape(value['latentModel'].getLatentSemantics(), (1,-1))
    #sim = similarity.cosine_similarity(flattend_target_latent_model,flattend_target_latent_pool_model)
    sim = np.dot(flattend_target_latent_model,flattend_target_latent_pool_model.T)
    simPair.append((key,sim[0][0]))

simPair = sorted(simPair,key = lambda x: x[1],reverse=True)
print("Top 3 Subject ID: ",simPair[0][0],simPair[1][0],simPair[2][0])

image1 = cv2.imread(imagePath+"/"+subject_img_dic_pool[simPair[0][0]]['image_list'][0]+'.jpg')
image2 = cv2.imread(imagePath+"/"+subject_img_dic_pool[simPair[1][0]]['image_list'][0]+'.jpg')
image3 = cv2.imread(imagePath+"/"+subject_img_dic_pool[simPair[2][0]]['image_list'][0]+'.jpg')

cv2.imshow("top1",image1)
cv2.imshow("top2",image2)
cv2.imshow("top3",image3)

cv2.waitKey(0)
cv2.destroyAllWindows()