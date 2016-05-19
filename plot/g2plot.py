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

data = []

f = open('g2.txt', 'r')
for line in f:
    data.append(float(line))

x1 = [i for i in range(0, 51)]
x2 = [i for i in range(50, 0, -1)]
plt.scatter(x1, data[0: 51])
plt.scatter(x2, data[51: 102], color='r')
plt.show()
