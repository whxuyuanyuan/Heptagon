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


mat = constructMat(60)
mat = (nd.interpolation.rotate(mat, 90, reshape=False, cval=0.0) > 0.2).astype(float)

plt.imshow(mat)
plt.show()