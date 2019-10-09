from .model import Model
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class ColorMoments(Model):
    def __init__(self, widthOfWindow=100, heightOfWindow=100):
        self._widthOfWindow = widthOfWindow
        self._heightOfWindow = heightOfWindow

    def extractFeatures(self, img):
        # Convert to YUV first.
        imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Get the height / width / number of channels
        h = imgYUV.shape[0]
        w = imgYUV.shape[1]
        c = imgYUV.shape[2]

        # The number of windows
        numOfRows = int((h + 1) / self._heightOfWindow)
        numOfColumns = int((w + 1) / self._widthOfWindow)

        # The results we wanted.
        resultMean = np.zeros(shape=(numOfRows, numOfColumns, c))
        resultStd = np.zeros(shape=(numOfRows, numOfColumns, c))
        resultSkew = np.zeros(shape=(numOfRows, numOfColumns, c))

        for hIdx in range(numOfRows):
            # Find the end index of row window.
            hEndIdx = (
                ((hIdx + 1) * self._heightOfWindow) if hIdx < numOfRows - 1 else None
            )

            for wIdx in range(numOfColumns):
                # Find the end index of column window.
                wEndIdx = (
                    ((wIdx + 1) * self._widthOfWindow)
                    if wIdx < numOfColumns - 1
                    else None
                )

                window = imgYUV[
                    hIdx * self._heightOfWindow : hEndIdx,
                    wIdx * self._widthOfWindow : wEndIdx,
                    :,
                ]

                resultMean[hIdx, wIdx, :] = np.mean(window, axis=(0, 1))
                resultStd[hIdx, wIdx, :] = np.std(window, axis=(0, 1))
                # I tried scipy.stats.skew, but it seems the result is not correct.
                # Implement it by myself.
                resultSkew[hIdx, wIdx, 0] = np.sum(np.power(window[:, :, 0] - resultMean[hIdx, wIdx, 0], 3)) / (self._heightOfWindow * self._widthOfWindow)
                resultSkew[hIdx, wIdx, 1] = np.sum(np.power(window[:, :, 1] - resultMean[hIdx, wIdx, 1], 3)) / (self._heightOfWindow * self._widthOfWindow)
                resultSkew[hIdx, wIdx, 2] = np.sum(np.power(window[:, :, 2] - resultMean[hIdx, wIdx, 2], 3)) / (self._heightOfWindow * self._widthOfWindow)

                resultSkew[hIdx, wIdx, 0] = np.sign(resultSkew[hIdx, wIdx, 0]) * (np.abs(resultSkew[hIdx, wIdx, 0])) ** (1/3)
                resultSkew[hIdx, wIdx, 1] = np.sign(resultSkew[hIdx, wIdx, 1]) * (np.abs(resultSkew[hIdx, wIdx, 1])) ** (1/3)
                resultSkew[hIdx, wIdx, 2] = np.sign(resultSkew[hIdx, wIdx, 2]) * (np.abs(resultSkew[hIdx, wIdx, 2])) ** (1/3)

        return (resultMean, resultStd, resultSkew)

    def getSimilarity(self, feature1, feature2, distanceFunction):
        # Calculate distance.
        mean1, std1, skew1 = feature1
        mean2, std2, skew2 = feature2

        # Make the features to one-dimension vectors.
        feature1 = np.concatenate((mean1, std1, skew1), axis=2).flatten()
        feature2 = np.concatenate((mean2, std2, skew2), axis=2).flatten()

        return distanceFunction(feature1, feature2)

    def getSimilarityScore(self, data):
        return data

    def serializeFeature(self, featuresData):
        return featuresData

    def deserializeFeature(self, data):
        return data

    def extendChannel(self, channel):
        return np.reshape(
            np.repeat(channel, 3), (channel.shape[0], channel.shape[1], 3)
        )

    def visualizeFeatures(self, img, feature):
        # If we do not assign TKAgg, the fig.canvas.tostring_rgb() would raise error
        # since MacOS version does not implement it.
        matplotlib.use("TKAgg", force=True)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Print the matrix of mean, std and skew
        with np.printoptions(precision=3, suppress=True):
            print("The mean of Y, U and V channel are:")
            print(feature[0][:, :, 0])
            print(feature[0][:, :, 1])
            print(feature[0][:, :, 2])

            print("The std of Y, U and V channel are:")
            print(feature[1][:, :, 0])
            print(feature[1][:, :, 1])
            print(feature[1][:, :, 2])

            print("The skew of Y, U and V channel are:")
            print(feature[2][:, :, 0])
            print(feature[2][:, :, 1])
            print(feature[2][:, :, 2])

        mean = feature[0] / 255.0
        std = feature[1] / np.max(feature[1], axis=(0, 1))

        skew = feature[2] - np.min(feature[2], axis=(0, 1))
        skew = skew / np.max(skew, axis=(0, 1))

        # https://stackoverflow.com/questions/25862026/turn-off-axes-in-subplots/25864515
        # https://stackoverflow.com/questions/37723963/broadcast-one-channel-in-numpy-array-into-three-channels
        fig, axarr = plt.subplots(4, 4)
        fig.set_size_inches(8, 8, forward=True)

        fig.suptitle("Color Moments")

        axarr[0, 0].imshow(imgRGB)
        axarr[0, 0].set_title("Origin image")
        axarr[0, 0].axis("off")

        axarr[0, 1].axis("off")
        axarr[0, 2].axis("off")
        axarr[0, 3].axis("off")

        # Generate all visualized result for each channel and each moment.
        channels = ["Y", "U", "V"]
        dataPair = [("Channel", imgYUV), ("mean", mean), ("std", std), ("skew", skew)]
        for i, channel in enumerate(channels):
            for j, data in enumerate(dataPair):
                axarr[i + 1, j].imshow(self.extendChannel(data[1][:, :, i]))
                axarr[i + 1, j].set_title(f"{channel} {data[0]}")
                axarr[i + 1, j].axis("off")

        # https://stackoverflow.com/questions/43099734/combining-cv2-imshow-with-matplotlib-plt-show-in-real-time
        fig.canvas.draw()

        # Convert to ndarray to that we can use opencv to process it.
        buf = fig.canvas.tostring_rgb()
        img = np.fromstring(buf, dtype=np.uint8, sep="").reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def visualizeSimilarityResult(
        self, img1, features1, img2, features2, similarityData, rank=0, score=0
    ):
        matplotlib.use("TKAgg", force=True)

        img1RGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1YUV = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)

        img2RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2YUV = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

        fig, axarr = plt.subplots(2, 4)
        fig.set_size_inches(8, 4, forward=True)
        fig.suptitle(f"Rank {rank} distance {score}")

        axarr[0, 0].imshow(img1RGB)
        axarr[0, 0].set_title("Target image")
        axarr[0, 0].axis("off")

        axarr[0, 1].imshow(self.extendChannel(img1YUV[:, :, 0]))
        axarr[0, 1].set_title("Y Channel")
        axarr[0, 1].axis("off")

        axarr[0, 2].imshow(self.extendChannel(img1YUV[:, :, 1]))
        axarr[0, 2].set_title("U Channel")
        axarr[0, 2].axis("off")

        axarr[0, 3].imshow(self.extendChannel(img1YUV[:, :, 2]))
        axarr[0, 3].set_title("V Channel")
        axarr[0, 3].axis("off")

        axarr[1, 0].imshow(img2RGB)
        axarr[1, 0].set_title("Query image")
        axarr[1, 0].axis("off")

        axarr[1, 1].imshow(self.extendChannel(img2YUV[:, :, 0]))
        axarr[1, 1].set_title("Y Channel")
        axarr[1, 1].axis("off")

        axarr[1, 2].imshow(self.extendChannel(img2YUV[:, :, 1]))
        axarr[1, 2].set_title("U Channel")
        axarr[1, 2].axis("off")

        axarr[1, 3].imshow(self.extendChannel(img2YUV[:, :, 2]))
        axarr[1, 3].set_title("V Channel")
        axarr[1, 3].axis("off")

        fig.canvas.draw()

        # Convert to ndarray to that we can use opencv to process it.
        buf = fig.canvas.tostring_rgb()
        img = np.fromstring(buf, dtype=np.uint8, sep="").reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def sortSimilarityScoreReverse(self):
        return False

    def dimensionReduction(self, featureList, dimRed, k=None):
        flatFeatureList = []

        if type(dimRed).__name__ == "LDA":
            print("LDA!!!!!!!!")
            pass
        else:
            for feature in featureList:
                flatFeature1 = np.reshape(feature[0], (1, -1))
                flatFeature2 = np.reshape(feature[1], (1, -1))
                flatFeature3 = np.reshape(feature[2], (1, -1))
                flatFeatures = np.concatenate((flatFeature1, flatFeature2, flatFeature3), axis=1)

                flatFeatureList.append(flatFeatures)

            featureMatrix = np.concatenate(flatFeatureList)

        return dimRed(featureMatrix, k)

        
