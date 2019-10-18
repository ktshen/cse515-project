import os
import glob
import cv2


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
            self.images.append(img)
            yield img_array.flatten() if self._flatten else img_array

    def get_parsed_image_list(self) -> list:
        return self.images
