import argparse
import numpy as np
from module.reader import RGBtoArrayReader
from module.DimRed import NMF
from sklearn.metrics.pairwise import cosine_similarity

# Get necessary arguments
parser = argparse.ArgumentParser(description="Phase 2 Task 7")
parser.add_argument("-k", "--topk", metavar="topk", type=int, help="K.", required=True)
parser.add_argument(
    "-img",
    "--image_dir",
    metavar="image_dir",
    type=str,
    help="The directory of the images",
    required=True
)
args = parser.parse_args()
topk = args.topk
image_directory = args.image_dir

# Get all the images from the directory and construct 1D array of each image
RAR = RGBtoArrayReader(image_directory, flatten=True)
flatten_images_array = np.array([array for array in RAR])

# Build the similarity matrix
similarity_matrix = cosine_similarity(flatten_images_array)
images = RAR.get_parsed_image_list()

# Perfrom NMF process and print results
nmf = NMF(topk, similarity_matrix)
nmf.printLatentSemantics(images, similarity_matrix)
