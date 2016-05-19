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

r0 = np.load('tmp/r0.npy')
for step in range(0, 10):
    imgWh = misc.imread('pictures/%4d_Wh.jpg' % step)

    cen_x = []
    cen_y = []
    orien = []

    f = open('ParticleData/data_%4d' % step, 'r')
    for line in f:
        data = [float(elem) for elem in line.split()]
        cen_x.append(data[0])
        cen_y.append(data[1])
        orien.append(data[2])
    plt.imshow(imgWh)
    for i in range(len(cen_x)):
        heptagon_plot(cen_x[i], cen_y[i], orien[i], r0)
    plt.savefig('movies/tmpPositionOnImage/postion_%4d.eps' % step, dpi=400)
    plt.close()