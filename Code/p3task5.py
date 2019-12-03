import argparse
import os
import numpy as np
from module.lsh import LSH
from module.database import FilesystemDatabase
from models import modelFactory
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
# from module.distanceFunction import distanceFunction

# np.set_printoptions(threshold=np.inf)

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

parser.add_argument(
    "-dir",
    "--imgdir",
    type=str,
    help="The image directory",
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
img_dir = args.imgdir

# Load database and create dataset
print("Loading files to dataset...")
db = FilesystemDatabase(f"{table}_{modelName}", create=False)
model = modelFactory.creatModel(modelName)
dataset = dict()

for image_id in db.keys():
    dataset[image_id] = model.flattenFecture(model.deserializeFeature(db.getData(image_id)))

# Get the number of total images that are considered
total_image_amount = len(dataset.keys())

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
candidates = lsh.get_t_most_similar_images(query_image, top_t)

# Save the result to "task5_output.pkl"
with open("task5_output.pkl", "wb") as fHndl:
    pickle.dump((query_image, candidates), fHndl)

# Check image directory exists
if not os.path.isdir(img_dir):
    print("Please input valid image directory")

# Initialize matplotlib figure instance
fig = plt.figure()


subplot_col = 3
subplot_row = (len(candidates)+1) // subplot_col + 1  if (len(candidates)+1) % subplot_col else (len(candidates)+1) // subplot_col

# Read image file and plot it on the figure
def plot_image(imagename, title, index):
    image_dir = os.path.join(img_dir, f"{imagename}.jpg")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Can't find the following file {image_dir}")
    a = fig.add_subplot(subplot_row, subplot_col, index+1)
    im = plt.imread(image_dir)
    plt.axis('off')
    plt.imshow(im)
    a.set_title(title, fontsize=8)


# Plot the target image first
plot_image(query_image, f"{query_image} Target", 0)

# Plot candidates to the image
for idx, candidate in enumerate(candidates):
    plot_image(candidate[0], f"{candidate[0]} {str(candidate[2])[:8]}", idx+1)

fig.suptitle(f"{top_t} Most Similar Images | Number of images considered: {total_image_amount} | Features: {dataset[query_image].shape[0]}", fontsize=10)
plt.show()


# Correct Answer for testing
# print("--------------------------")
# calculate_distance = distanceFunction.createDistance("l2")
# candidates_with_distance = []
# for key, value in dataset.items():
#     if key == query_image:
#         continue
#     candidates_with_distance.append([key, calculate_distance(dataset[query_image], value)])
#
# candidates_with_distance.sort(key=lambda x: x[1])
# for idx, row in enumerate(candidates_with_distance):
#     print("{} {} {}".format(idx, row[0], row[1]))
#     if idx == 20:
#         break
