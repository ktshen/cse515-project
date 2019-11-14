import argparse
import numpy as np
from module.lsh import LSH
from module.database import FilesystemDatabase
from models import modelFactory
from module.distanceFunction import distanceFunction
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description="Phase 3 Task 5")

parser.add_argument(
    "-l",
    "--L_layers",
    type=int,
    help="The number of layers.",
    required=True
)

parser.add_argument(
    "-k",
    "--k_hashes",
    type=int,
    help="The number of hashes per layer.",
    required=True
)

parser.add_argument(
    "-i",
    "--image_id",
    type=str,
    help="Query image ID.",
    required=True
)

parser.add_argument(
    "-t",
    "--top_t",
    type=int,
    help="Most t similar images.",
    required=True
)

parser.add_argument(
    "-tb",
    "--table",
    type=str,
    help="The table will be used.",
    required=True,
)

parser.add_argument(
    "-d",
    "--model",
    type=str,
    help="The method will be used to reduce dimension.",
    required=True,
)

# extract argument
args = parser.parse_args()

L_layers = args.L_layers
k_hashes = args.k_hashes
query_image = args.image_id
top_t = args.top_t
modelName = args.model.lower()
table = args.table.lower()

# Load database and create dataset
print("Loading files...")
db = FilesystemDatabase(f"{table}_{modelName}", create=False)
model = modelFactory.creatModel(modelName)
dataset = dict()

for image_id in db.keys():
    dataset[image_id] = model.flattenFecture(model.deserializeFeature(db.getData(image_id)))

# Scaling
print("Scaling dataset...")
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.asarray(list(dataset.values())))

for image_id in db.keys():
    dataset[image_id] = (scaler.transform(dataset[image_id].reshape(1, -1))).reshape(-1)

print(f"Using {modelName} model with {dataset[query_image].shape[0]} features")

# Initialize LSH instance
lsh = LSH(L_layers, k_hashes)
lsh.build_structure(dataset)

# Get result
lsh.get_t_most_similar_images(query_image, top_t)


# Correct Answer for testing
print("--------------------------")
calculate_distance = distanceFunction.createDistance("l2")
candidates_with_distance = []
for key, value in dataset.items():
    if key == query_image:
        continue
    candidates_with_distance.append([key, calculate_distance(dataset[query_image], value)])

candidates_with_distance.sort(key=lambda x: x[1])
for idx, row in enumerate(candidates_with_distance):
    print("{} {} {}".format(idx, row[0], row[1]))
    if idx == 20:
        break
