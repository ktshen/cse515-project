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

parser = argparse.ArgumentParser(description="Phase 2 Task 7")
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
    "-meta",
    "--metadata",
    metavar="METADATA_PATH",
    type=str,
    help="Path of metadata.",
    required=True
)

args = parser.parse_args()

# extract argument
table = args.table.lower()
k = args.topk
metadataPath = Path(args.metadata)

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
model = modelFactory.creatModel('cm')

# The imageIdList and featureList should could be mapped to each other.
subject_img_dic_pool = {}



for key in db.keys():
    subject_id = imgToSub.get(key)
    if subject_img_dic_pool.get(subject_id) is None:
        subject_img_dic_pool[subject_id] = {
            'image_list' : [key],
            'flatten_feature':[],
            'latentModel':""
        }
    else:
        (subject_img_dic_pool.get(subject_id)['image_list']).append(key)


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



sub_sub_sim_matrix = np.zeros((len(subject_img_dic_pool.keys()),len(subject_img_dic_pool.keys())))


for key,value in subject_img_dic_pool.items():
    value['latentModel'] = DimRed.createReduction(decompMethod, k= smallest_k, data=value['flatten_feature'] )
    flattend_target_latent_pool_model = np.reshape(value['latentModel'].getLatentSemantics(), (1,-1))


for i, (key,value) in enumerate(subject_img_dic_pool.items()):
    for j, (key_1, value_1) in enumerate(subject_img_dic_pool.items()):
        #sim = similarity.cosine_similarity(flattend_target_latent_model,flattend_target_latent_pool_model)
        sub_sub_sim_matrix[i][j] = np.dot( np.reshape(value['latentModel'].getLatentSemantics(), (1,-1)), np.reshape(value_1['latentModel'].getLatentSemantics(), (1,-1)).T)




nmf_obj = DimRed.createReduction("nmf", k= k, data=sub_sub_sim_matrix )
nmf_obj.printLatentSemantics(list(subject_img_dic_pool.keys()), sub_sub_sim_matrix, "")