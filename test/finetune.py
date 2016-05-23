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

def heptagon_plot(I1, I2, A0, r0):
    plt.plot([I2 + r0 * m.cos(A0 + 0 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 1 * 2 * m.pi / 7),
              I2 + r0 * m.cos(A0 + 2 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 3 * 2 * m.pi / 7),
              I2 + r0 * m.cos(A0 + 4 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 5 * 2 * m.pi / 7),
              I2 + r0 * m.cos(A0 + 6 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 7 * 2 * m.pi / 7)],
             [I1 + r0 * m.sin(A0 + 0 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 1 * 2 * m.pi / 7),
              I1 + r0 * m.sin(A0 + 2 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 3 * 2 * m.pi / 7),
              I1 + r0 * m.sin(A0 + 4 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 5 * 2 * m.pi / 7),
              I1 + r0 * m.sin(A0 + 6 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 7 * 2 * m.pi / 7)], linewidth=0.3,
             color='black')



def InsidePolygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].
    :param x:
    :param y:
    :param points:
    :return:
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if (p1x == p2x) or (x <= xinters):
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def constructMat(r):
    Mat = np.zeros([2 * int(r * 1.), 2 * int(r * 1.)])
    q1 = r + np.array([r * m.cos(0 * 2 * m.pi / 7), r * m.sin(0 * 2 * m.pi / 7)])
    q2 = r + np.array([r * m.cos(1 * 2 * m.pi / 7), r * m.sin(1 * 2 * m.pi / 7)])
    q3 = r + np.array([r * m.cos(2 * 2 * m.pi / 7), r * m.sin(2 * 2 * m.pi / 7)])
    q4 = r + np.array([r * m.cos(3 * 2 * m.pi / 7), r * m.sin(3 * 2 * m.pi / 7)])
    q5 = r + np.array([r * m.cos(4 * 2 * m.pi / 7), r * m.sin(4 * 2 * m.pi / 7)])
    q6 = r + np.array([r * m.cos(5 * 2 * m.pi / 7), r * m.sin(5 * 2 * m.pi / 7)])
    q7 = r + np.array([r * m.cos(6 * 2 * m.pi / 7), r * m.sin(6 * 2 * m.pi / 7)])

    for it0 in range(0, Mat.shape[0]):
        for it1 in range(0, Mat.shape[1]):
            if InsidePolygon(it1 + 1, it0 + 1, [q1, q2, q3, q4, q5, q6, q7]):
                Mat[it0, it1] = 1

    return Mat

def OverlapVal(x):
    global MatHept, MatHeptEdge, subMap1, subMap2, r0, angle
    dx = x[0]
    dy = x[1]
    dA = x[2]
    A = angle + dA
    val = 100
    if (abs(dx) < int(0.05 * r0)) and (abs(dy) < int(0.05 * r0) and abs(dA) < 0.05):
        try:
            # rotate pattern matrix
            MatHeptRot = (nd.interpolation.rotate(MatHept, A/m.pi*180, reshape=False, cval=0.0) > 0.2).astype(float)
            MatHeptEdgeRot = (nd.interpolation.rotate(MatHeptEdge, A/m.pi*180, reshape=False, cval=0.0) > 0.2).astype(float)
            # compute overlap value

            val1 = (np.sum(MatHeptRot * subMap1[
                                       subMap1.shape[0] / 2 + dx - MatHeptRot.shape[0] / 2:subMap1.shape[0] / 2 + dx +
                                                                                          MatHeptRot.shape[0] / 2,
                                       subMap1.shape[1] / 2 + dy - MatHeptRot.shape[1] / 2:subMap1.shape[1] / 2 + dy +
                                                                                          MatHeptRot.shape[
                                                                                               1] / 2]) / np.sum(
                MatHeptRot))

            val2 = (np.sum(MatHeptEdgeRot * subMap2[subMap2.shape[0] / 2 + dx - MatHeptEdgeRot.shape[0] /
                                                                                2:subMap2.shape[0] / 2 + dx +
                                                                                  MatHeptEdgeRot.shape[0] / 2,
                                            subMap2.shape[1] / 2 + dy - MatHeptEdgeRot.shape[1] / 2:subMap2.shape[1] / 2
                                                                                                    + dy +
                                                                                                    MatHeptEdgeRot.shape[1]
                                                                                                    / 2]) /
                    np.sum(MatHeptEdgeRot))
            val = 0.4/val1 + 0.6/val2
        except:
            val = 100
    return val



imgUv = misc.imread('uv.JPG')
imgWh = misc.imread('Wh.JPG')
imgWh = imgWh[:, :, 0]  # extract red part
plt.imshow(imgWh)

cen_x = []
cen_y = []
orien = []

f = open('centers.txt', 'r')
for line in f:
    data = [float(elem) for elem in line.split()]
    cen_y.append(data[0])
    cen_x.append(data[1])
    orien.append(data[2])
"""
###################### radius ###################################
plt.imshow(imgWh)
plt.title('zoom and click on 7 vertices of a heptagon, then middle click and close')
_, q1, q2, q3, q4, q5, q6, q7 = ginput(0, 0)
plt.show()
plt.close()

r0 = (0.5 / m.sin(m.pi / 7)) * np.mean(np.array(
    [m.sqrt((q1[0] - q2[0]) ** 2 + (q1[1] - q2[1]) ** 2), m.sqrt((q2[0] - q3[0]) ** 2 + (q2[1] - q3[1]) ** 2),
     m.sqrt((q3[0] - q4[0]) ** 2 + (q3[1] - q4[1]) ** 2), m.sqrt((q4[0] - q5[0]) ** 2 + (q4[1] - q5[1]) ** 2),
     m.sqrt((q5[0] - q6[0]) ** 2 + (q5[1] - q6[1]) ** 2), m.sqrt((q6[0] - q7[0]) ** 2 + (q6[1] - q7[1]) ** 2),
     m.sqrt((q7[0] - q1[0]) ** 2 + (q7[1] - q1[1]) ** 2)]))
"""
r0 = 69
MatHept = constructMat(r0)

# Make the pattern of edges:
MatHeptEdge = cv2.Canny(1 - MatHept.astype('uint8'), 0, 1)
kernel = np.ones((int(MatHeptEdge.shape[0] * 0.008) * 2 + 1, int(MatHeptEdge.shape[0] * 0.008) * 2 + 1), np.uint8)
MatPentEdge = cv2.dilate(MatHeptEdge, kernel, iterations=1)


####################################################################

for i in range(len(cen_x)):
    x1 = cen_x[i]
    y1 = cen_y[i]
    angle = orien[i]
    # Extract interesting sub-part of the matrix
    subMap = imgWh[x1 - int(1.1*r0):x1 + int(1.1*r0), y1 - int(1.1*r0):y1 + int(1.1*r0)]
    # Threshold the the sub-matrix:
    #subMap = cv2.GaussianBlur(subMap.astype('uint8'),
    #                       (int(subMap.shape[0] * 0.02 / 2) * 2 + 1, int(subMap.shape[0] * 0.02 / 2) * 2 + 1), 0)
    subMap = subMap.astype('uint8')
    subMap1 = (subMap < 180).astype(int) * (subMap > 100).astype(int)
    subMap2 = cv2.adaptiveThreshold(subMap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    subMap2 = (subMap2 < 120).astype(int)
    guess = [0.0, 0.0, 0.0]
    optim = optimize.minimize(OverlapVal, guess, method='Powell', options={'disp': False, 'maxfev': 150, 'ftol': 10 ** (-5), 'xtol': 10 ** (-5)})
    print optim.fun
    cen_x[i] += optim.x[0]
    cen_y[i] += optim.x[1]
    orien[i] += optim.x[2]


imgWh = misc.imread('Wh.JPG')

plt.imshow(imgWh)
for i in range(len(cen_x)):
    heptagon_plot(cen_x[i], cen_y[i], orien[i], r0)
plt.savefig('finetune_test.eps', dpi=400)
plt.show()
