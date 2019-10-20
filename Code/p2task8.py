import argparse
import numpy as np
from module.reader import RGBtoArrayReader, get_image_array_with_metadata
from module.DimRed import DimRed
from sklearn.metrics.pairwise import cosine_similarity

# Get necessary arguments
parser = argparse.ArgumentParser(description="Phase 2 Task 8")
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=True)
parser.add_argument(
    "-img",
    "--image_dir",
    metavar="image_dir",
    type=str,
    help="The directory of the images",
    required=True
)
parser.add_argument(
    "-meta",
    "--meta_dir",
    metavar="meta_dir",
    type=str,
    help="The directory of the images",
    required=True
)
args = parser.parse_args()
topk = args.topk
image_directory = args.image_dir
meta_directory = args.meta_dir

# Construct matrix with image and meta space combined
print(f"Reading image files")
RAR = RGBtoArrayReader(image_directory, flatten=True)
print(f"Flatten the image data and generate one-hot vector")
images, flatten_images_array, img_shape = get_image_array_with_metadata(meta_directory, RAR)

# Perform NMF process
print("Processing NMF, please wait")
nmf = DimRed.createReduction('nmf', k=topk, data=flatten_images_array)

print("Top-k latent semantics in the image-space -> (order, term)")
for order, array in enumerate(nmf.components_):
    print((order+1, array[:img_shape[0]]))

print("Top-k latent semantics in the metadata-space -> (order, term)")
for order, array in enumerate(nmf.components_):
    print((order+1, array[img_shape[0]:]))
