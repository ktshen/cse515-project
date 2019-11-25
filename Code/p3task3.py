import argparse
import os
import numpy as np
from pathlib import Path
from module.DimRed import DimRed
from models import modelFactory
from module.database import FilesystemDatabase
from classifier.classifier import Classifier


#python p3task3.py -k 5 -lk 10 -t test -i phase3_sample_data/sample/ -lst Hand_0003457.jpg,Hand_0000074.jpg,Hand_0005661.jpg
#python p3task3.py -k 5 -lk 10 -t test -i phase3_sample_data/sample/ -lst Hand_0008333.jpg,Hand_0006183.jpg,Hand_0000074.jpg


parser = argparse.ArgumentParser(description="Phase 3 Task 3")

parser.add_argument(
    "-k",
    type=int,
    help="k outgoing edges",
    required=True
)

parser.add_argument(
    "-lk",
    type=int,
    help="Most K dominant images",
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
    "-i",
    "--input_path",
    metavar="input_path",
    type=str,
    help="Input folder.",
    required=True,
)


parser.add_argument(
    "-lst",
    type=str,
    help="Three query image IDs",
    required=True,
)

args = parser.parse_args()
k = args.k
lk = args.lk
inputPath = Path(args.input_path)
query_img_list = args.lst.split(',')

table = args.table.lower()


# computing latent semantics and projection
decompMethod = 'svd'
usedModels = "hog"
  
db = FilesystemDatabase(f"{table}_"+usedModels, create=False)
model = modelFactory.creatModel(usedModels)

SUPPORT_FILE_TYPES = [".jpg"]


image_list = []
image_feature_matrix = []




for fileName in os.listdir(inputPath):
    for extension in SUPPORT_FILE_TYPES:
        if fileName.endswith(extension):
            image_list.append(fileName)
            image_feature_matrix. append(
                model.flattenFecture(model.deserializeFeature(db.getData(inputPath / fileName)), decompMethod)
            )

image_simlarity_dict = {}

for i,feature in enumerate(image_feature_matrix):
    img_img_sim = []
    for feature_p in image_feature_matrix:
        img_img_sim.append(np.dot(feature,feature_p.T))

    a = np.array(img_img_sim)
    ind =a.argsort()[-k:][::-1]

    image_simlarity_dict[image_list[i]]={
        "sim_weight": a[ind],
        "sim_node_index":ind
    }



classifier = Classifier.createClassifier("ppr",**{"img_sim_graph":image_simlarity_dict,"image_list":image_list,"lk":lk})

print(np.array(image_list)[classifier.get_steady_state(query_img_list)])

















