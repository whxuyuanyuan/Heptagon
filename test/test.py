# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from pylab import ginput
import cv2
import cv2.cv as cv
import os

def heptagon_plot(I2, I1, A0, r0):
    plt.plot([I2 + r0 * m.cos(A0 + 0 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 1 * 2 * m.pi / 7),
              I2 + r0 * m.cos(A0 + 2 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 3 * 2 * m.pi / 7),
              I2 + r0 * m.cos(A0 + 4 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 5 * 2 * m.pi / 7),
              I2 + r0 * m.cos(A0 + 6 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 7 * 2 * m.pi / 7)],
             [I1 + r0 * m.sin(A0 + 0 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 1 * 2 * m.pi / 7),
              I1 + r0 * m.sin(A0 + 2 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 3 * 2 * m.pi / 7),
              I1 + r0 * m.sin(A0 + 4 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 5 * 2 * m.pi / 7),
              I1 + r0 * m.sin(A0 + 6 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 7 * 2 * m.pi / 7)], linewidth=0.8,
             color='black')


imgUv = misc.imread('pictures/IMG_0016.JPG')
imgWh = misc.imread('pictures/IMG_0017.JPG')
imgPl = misc.imread('pictures/IMG_0018.JPG')


imgWh = imgWh[554: 1366, 1452: 2066]
imgUv = imgUv[554: 1366, 1452: 2066]
imgPl = imgPl[554: 1366, 1452: 2066]

# Extract the red and blue parts respectively
#imgWh = imgWh[:, :, 0]
imgUv = imgUv[:, :, 2]



thUv = -15

# Binarize the image:
# Blur picture:
imgUvBin = cv2.GaussianBlur(imgUv.astype('uint8'), (7, 7), 0)
imgUvBin = cv2.adaptiveThreshold(imgUvBin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, thUv)

imgUvLabel = nd.measurements.label(imgUvBin)[0]

centers = nd.measurements.center_of_mass(imgUvBin, imgUvLabel, range(1, int(np.amax(imgUvLabel) + 1)))
areas = nd.measurements.sum(imgUvBin, imgUvLabel, range(1, int(np.amax(imgUvLabel) + 1)))

print areas

centers_temp = []

for i in range(len(centers)):
    if areas[i] > 100000:
        centers_temp.append(centers[i])
centers = centers_temp
plt.imshow(imgUvBin)

for i in range(len(centers)):
    plt.scatter(centers[i][1], centers[i][0])
plt.show()



plt.imshow(imgWh)


# Hough Circle Transform
circles = cv2.HoughCircles(imgUv, cv.CV_HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=25, minRadius=0, maxRadius=0)

x = []
y = []
orien = []

for i in circles[0, :]:
    if 10 < i[2] < 20:
        plt.scatter(i[0], i[1], color='r')
        x.append(i[0])
        y.append(i[1])
        minVal = 10000
        index_temp = 1
        for j in range(len(centers)):
            dist2 = (centers[j][1] - x[-1]) ** 2 + (centers[j][0] - y[-1]) ** 2
            if 1000 < dist2 < minVal:
                minVal = dist2
                index_temp = j
        orien.append(np.arctan2((centers[index_temp][0] - y[-1]), centers[index_temp][1] - x[-1]))
        # plot
        plt.plot([centers[index_temp][1], x[-1]], [centers[index_temp][0], y[-1]])
        heptagon_plot(x[-1], y[-1], orien[-1], 70)


plt.show()
