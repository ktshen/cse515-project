from .model import Model
import cv2
import numpy as np
import math


class SIFT(Model):
    KEYPOINTS = "keypoints"
    DESCRIPTORS = "descriptors"

    def __init__(self):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._bfmatcher = cv2.BFMatcher()

    def extractFeatures(self, img):
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keyPoints, descriptor = self._sift.detectAndCompute(grayImg, None)

        return {SIFT.KEYPOINTS: keyPoints, SIFT.DESCRIPTORS: descriptor}

    # Since KeyPoints cannot be automatically serialized by Python.
    # We have to convert it by ourself.
    # https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
    # https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    def serializeFeature(self, data):
        keyPoints, descriptors = data[SIFT.KEYPOINTS], data[SIFT.DESCRIPTORS]
        serializedList = []

        for i, keyPoint in enumerate(keyPoints):
            serializedList.append(
                (
                    keyPoint.pt,
                    keyPoint.size,
                    keyPoint.angle,
                    keyPoint.response,
                    keyPoint.octave,
                    keyPoint.class_id,
                )
            )

        serializedList.append(descriptors)

        return serializedList

    def deserializeFeature(self, data):
        keyPoints = []
        descriptors = data.pop(-1)

        for point in data:
            keyPoints.append(
                cv2.KeyPoint(
                    x=point[0][0],
                    y=point[0][1],
                    _size=point[1],
                    _angle=point[2],
                    _response=point[3],
                    _octave=point[4],
                    _class_id=point[5],
                )
            )

        return {SIFT.KEYPOINTS: keyPoints, SIFT.DESCRIPTORS: descriptors}

    def getSimilarity(self, data1, data2, distanceFunction):
        desc1 = data1[SIFT.DESCRIPTORS]
        desc2 = data2[SIFT.DESCRIPTORS]

        # This is match function of opencv, used for verification.
        # matches = self._bfmatcher.knnMatch(desc1, desc2, k=2)

        matches = []

        # Check rows of two images pairwisely.
        for i in range(len(desc1)):
            leastDistance = None
            secondLeast = None

            for j in range(len(desc2)):
                dist = distanceFunction(desc1[i], desc2[j])

                # Find 2 points with least distances
                if leastDistance is None:
                    leastDistance = (dist, j)
                elif dist < leastDistance[0]:
                    secondLeast = leastDistance
                    leastDistance = (dist, j)
                else:
                    if secondLeast is None:
                        secondLeast = (dist, j)
                    elif dist < secondLeast[0]:
                        secondLeast = (dist, j)

            # Make a DMatch points list so that we can draw line by OpenCV.
            matches.append(
                [
                    cv2.DMatch(i, leastDistance[1], leastDistance[0]),
                    cv2.DMatch(i, secondLeast[1], secondLeast[0]),
                ]
            )

        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])

        return good

    def getSimilarityScore(self, data):
        return len(data)

    def visualizeFeatures(self, img, feature):
        return cv2.drawKeypoints(
            img,
            feature[SIFT.KEYPOINTS],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    def visualizeSimilarityResult(
        self, img1, features1, img2, features2, matchesData, rank=0, score=0
    ):
        # https://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
        matchingImg = cv2.drawMatchesKnn(
            img1,
            features1[SIFT.KEYPOINTS],
            img2,
            features2[SIFT.KEYPOINTS],
            matchesData,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        """
        matplotlib.use("TKAgg", force=True)
        imgRGB = cv2.cvtColor(matchingImg, cv2.COLOR_BGR2RGB)

        fig, axarr = plt.subplots(1, 1)
        fig.set_size_inches(10, 6, forward=True)
        fig.suptitle(f"Rank {rank} distance {score}")

        axarr.imshow(imgRGB, aspect="auto")
        axarr.axis("off")

        fig.canvas.draw()

        # Convert to ndarray to that we can use opencv to process it.
        buf = fig.canvas.tostring_rgb()
        img = np.fromstring(buf, dtype=np.uint8, sep="").reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        """

        return matchingImg

    def sortSimilarityScoreReverse(self):
        return True

    def dimensionReduction(self, featureList, dimRed, k=None):
        ffList = []
        for i, feature in enumerate(featureList):
            ffList.append(self.flattenFecture(feature))

        finalMatrix = np.array(ffList).reshape(-1, 192 * 128)

        return dimRed(finalMatrix)

    def flattenFecture(self, feature, dimRedName=None):
        finalList = []
        temp_list = np.array(np.zeros((12, 16)), dtype=list)
        for i in range(0, 12):
            for j in range(0, 16):
                temp_list[i][j] = [np.zeros((128,)), 0]

        for kp, des in zip(feature["keypoints"], feature["descriptors"]):
            cord_x = math.floor(kp.pt[0] / 100)
            cord_y = math.floor(kp.pt[1] / 100)
            temp_list[cord_y, cord_x][0] += np.array(des)
            temp_list[cord_y, cord_x][1] += 1
        for i in range(12):
            for j in range(16):
                if temp_list[i][j][1] == 0:
                    finalList.append(temp_list[i][j][0])
                else:
                    finalList.append(temp_list[i][j][0] / temp_list[i][j][1])

        return np.array(finalList).reshape((-1))
