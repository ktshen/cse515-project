import os
import glob
import cv2
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class RGBtoArrayReader:
    def __init__(self, path: str, ext='.jpg', normalized=True, flatten=True):
        self._dir = path
        self._ext = ext.strip(".")
        self._normalized = normalized
        self._flatten = flatten
        self.images = []

        if not os.path.isdir(self._dir):
            raise NotADirectoryError(f"{self._dir} is not found or does not have permission to access")

    def __iter__(self):
        for img in glob.glob(os.path.join(self._dir, f"*.{self._ext}")):
            img_array = cv2.imread(img, cv2.COLOR_BGR2RGB)
            if self._normalized:
                img_array = cv2.normalize(img_array, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            self.images.append(img.split("/")[-1].strip('.jpg'))
            yield img_array.flatten() if self._flatten else img_array

    def get_parsed_image_list(self) -> list:
        return self.images

    def get_last_image_file(self) -> str:
        return self.images[-1]


def get_image_array_with_metadata(path: str, rgb_reader: RGBtoArrayReader) -> [list, np.ndarray, tuple]:
    if not os.path.exists(path):
        raise NotADirectoryError(f"{path} is not found or does not have permission to access")

    # key: image filename, value: index in meta_onehot
    image2metaidx = {}
    meta = []

    with open(path, "r") as metadataFile:
        csvReader = csv.reader(metadataFile, delimiter=',')

        # skip the headers
        next(csvReader, None)

        index = 0
        for row in csvReader:
            image2metaidx[row[7].strip('.jpg')] = index
            # Category :gender, accessories, nailPolish, aspectOfHand
            meta.append(row[2:3]+row[4:7])
            index += 1

    # Using one-hot encoding to transform the metadata
    enc = OneHotEncoder(handle_unknown='ignore')
    meta_onehot = enc.fit_transform(meta).toarray()

    # Append metadata to RGB flatten array
    images = rgb_reader.get_parsed_image_list()
    flatten_images_array = np.array([array for array in rgb_reader])
    img_shape = flatten_images_array[0].shape
    concatenate_array = []
    for idx in range(len(images)):
        concatenate_array.append(np.hstack([flatten_images_array[idx], meta_onehot[image2metaidx[images[idx]]]]))

    return [images, np.array(concatenate_array), img_shape]
