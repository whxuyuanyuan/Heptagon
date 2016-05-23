# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from pylab import ginput
import cv2
import os

axis1 = 38.60
axis2 = 38.32
stepsize = 0.04
particleArea = 618.0 * 7.0 / 4.0 * 0.6934**2 / np.tan(np.pi / 7)
phi = [particleArea / (axis1 * axis2)]

for i in range(1, 51):
    axis1 -= stepsize
    axis2 -= stepsize
    phi.append(particleArea / (axis1 * axis2))

for i in range(49, -1, -1):
    phi.append(phi[i])

ave_con = []
g2 = []
fileInput = open('g2.txt')
for line in fileInput:
    g2.append(float(line))

fileInput = open('average_contact_number.txt')
for line in fileInput:
    ave_con.append(float(line))


fig, ax1 = plt.subplots()
ax1.plot(phi[0: 51], g2[0: 51], 'b.-', label='Compression')
ax1.plot(phi[50: 101], g2[50: 101], 'bx-', markersize=5, label='Decompression')
ax1.set_xlabel('packing fraction ' + r'$\phi$')
ax1.set_ylabel(r'$G^{2}$', color='b')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
for tl in ax1.get_yticklabels():
    tl.set_color('b')


ax2 = ax1.twinx()
ax2.plot(phi[1: 51], ave_con[1: 51], 'r.-', label='Compression')
ax2.plot(phi[50: 101], ave_con[50: 101], 'rx-', markersize=5, label='Decompression')

ax2.set_ylabel('Contact number ' + r'$Z$', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
ax1.legend(loc='lower right')
ax2.legend(loc='upper left')
plt.savefig('compare.eps')
